---
title: >-
  [Paper Note] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting
description: >-
  [AAAI 2026][Time Series][Time series forecasting] This paper proposes the FreqCycle framework, which explicitly learns shared periodic patterns via the FECF module…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Time series forecasting"
  - "frequency-domain analysis"
  - "periodicity modeling"
  - "mid-to-high frequency enhancement"
  - "multi-scale decomposition"
date: 2026-05-08
content_hash: a1de66ea46856d14
---

# FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting

**Conference**: AAAI 2026
**arXiv**: [2603.09661](https://arxiv.org/abs/2603.09661)  
**Code**: [github.com/boya-zhang-ai/FreqCycle](https://github.com/boya-zhang-ai/FreqCycle)  
**Area**: Time Series Forecasting
**Keywords**: Time series forecasting, frequency-domain analysis, periodicity modeling, mid-to-high frequency enhancement, multi-scale decomposition

## TL;DR

This paper proposes the FreqCycle framework, which explicitly learns shared periodic patterns via the FECF module, enhances mid-to-high frequency energy contributions via the SFPL module, and extends to MFreqCycle for handling coupled multi-periodicity. The framework achieves an optimal balance of SOTA performance and efficiency across 7 benchmarks.

## Background & Motivation

Mining time-frequency features is critical in time series forecasting (TSF). Existing research exhibits three key shortcomings:

**Over-reliance on complex architectures for periodic pattern extraction**: Deep learning models attempt to capture periodic regularities in long-range dependencies through complex structures such as attention mechanisms, yet many time series (e.g., electricity, traffic) exhibit clear daily and weekly periodic patterns (as illustrated in Figure 1 for the ETTm2 dataset), which can be modeled more directly and explicitly.

**Neglect of mid-to-high frequency components**: Existing methods (e.g., DLinear and other MLP-based models) are effective at learning low-frequency periodic components, but their fundamental architectures are inherently limited in representing mid-to-high frequency components. Spectral analysis shows that the energy contribution of mid-to-high frequency components is extremely low (Figure 2, left), yet these components carry critical information about short-term fluctuations and non-periodic features.

**Lack of dedicated treatment for multi-scale nested periodicity**: Time series data frequently exhibit coupled multi-periodicity (e.g., daily cycles nested within weekly cycles), which simple MLP models cannot adaptively model across multiple hierarchical levels.

## Method

### Overall Architecture

FreqCycle (Figure 3a) consists of two complementary modules:
1. **FECF (Filter-Enhanced Cycle Forecasting)**: Explicitly learns shared periodic patterns in the time domain, handling low-frequency components.
2. **SFPL (Segmented Frequency-domain Pattern Learning)**: Enhances mid-to-high frequency energy contributions in the frequency domain, handling non-periodic components in the residual.

Overall pipeline: Input → FECF extracts and removes periodic components → Residual fed into SFPL → Residual prediction + filtered periodic prediction = Final prediction.

For long lookback window scenarios, the framework is further extended to **MFreqCycle**, which decouples nested periodic features via a multi-scale parallel architecture.

### Key Designs

#### 1. **FECF (Filter-Enhanced Cycle Forecasting) Module**

FECF is an improvement over CycleNet. Its core idea is to **explicitly learn a globally shared periodic basis**:

- Define a learnable periodic basis $Q \in \mathbb{R}^{W \times D}$ (initialized to a zero matrix), where $W$ is the base period length.
- Generate periodic components $c_{t-L+1:t}$ and $c_{t+1:t+H}$ of lengths matching the input and output via periodic replication.
- **Adaptive filter enhancement**: Apply frequency-domain filtering to the predicted periodic component to amplify low-frequency information and attenuate mid-to-high frequency interference:

$$c'_{t+1:t+H} = \text{IFFT}(\text{Filter}(\text{FFT}(c_{t+1:t+H})))$$

$$\text{Filter}(\xi) = \xi \odot \theta_c$$

where $\theta_c$ is a learnable filter parameter and $\odot$ denotes element-wise multiplication.

Complete FECF pipeline:
1. Extract residual: $r_{t-L+1:t} = x_{t-L+1:t} - c_{t-L+1:t}$
2. Residual prediction: $r_{t+1:t+H} = \text{SFPL}(r_{t-L+1:t})$
3. Reconstruct prediction: $\bar{x}_{t+1:t+H} = r_{t+1:t+H} + c'_{t+1:t+H}$

The base period $W$ is determined by the intrinsic properties of the data: $W=24$ (daily) or $W=168$ (weekly) for hourly-sampled data.

#### 2. **SFPL (Segmented Frequency-domain Pattern Learning) Module**

SFPL is specifically designed to **enhance the energy contribution of mid-to-high frequency components**, inspired by the Short-Time Fourier Transform (STFT):

1. **Segmentation**: Divide the input $r_{t-L+1:t}$ into $s$ sub-segments via a sliding window; zero-pad both ends and stack to obtain $R \in \mathbb{R}^{s \times L \times D}$.
2. **Frequency-domain transformation**: Apply FFT along the time dimension of $R$ to obtain $F \in \mathbb{C}^{s \times \lfloor L/2+1 \rfloor \times D}$.
3. **Learnable filtering and adaptive weighting**:

$$\theta'_1, \ldots, \theta'_s = \text{softmax}(\theta_1, \ldots, \theta_s)$$
$$F' = F \odot \Theta'_F, \quad f = \sum_{i=1}^{s} F'(i)$$

4. **Reconstruction**: $f$ is passed through iFFT and an FFN layer to produce the residual prediction output.

The segmentation operation is effectively a variant of STFT: shortening the time-domain window achieves higher frequency-domain locality, enabling more precise localization of transient frequency components. The frequency-domain representation $f$, obtained after learnable filtering and adaptive weighted aggregation, effectively enhances mid-to-high frequency energy (Figure 2, right vs. left).

#### 3. **MFreqCycle (Multi-Scale Extension)**

To handle coupled multi-periodicity (e.g., daily–weekly nesting), MFreqCycle adopts a multi-scale parallel architecture:

- **Base-period module**: Captures the smallest significant period (e.g., daily) and its non-periodic features.
- **Weekly-period module**: Models macro-scale periodic patterns (e.g., weekly), using a longer input window.
    - Comprises a pooling layer + feature learning module + Linear layer.
    - The pooling and projection design extracts key trends while maintaining computational efficiency.

Multi-scale prediction fusion:

$$\theta'_0, \theta'_1 = \text{softmax}(\theta_0, \theta_1)$$
$$\bar{x}_{t+1:t+H} = \bar{x}^0_{t+1:t+H} \odot \theta'_0 + \bar{x}^1_{t+1:t+H} \odot \theta'_1$$

### Loss & Training

- Standard MSE loss is used for end-to-end training.
- The periodic basis $Q$ and adaptive filters are jointly trained via gradient backpropagation.
- Theoretical complexity is $O(L \log L)$, avoiding the quadratic complexity of attention mechanisms.

## Key Experimental Results

### Main Results (7 datasets, L=96, H∈{96,192,336,720}, averaged)

| Dataset | FreqCycle (MSE/MAE) | CycleNet | DLinear | iTransformer | PatchTST |
|--------|-------------------|----------|---------|-------------|----------|
| ETTm1 | **0.372/0.389** | 0.386/0.395 | 0.403/0.407 | 0.407/0.410 | 0.387/0.400 |
| ETTm2 | **0.263/0.311** | 0.272/0.315 | 0.350/0.401 | 0.288/0.332 | 0.281/0.326 |
| ETTh1 | **0.428/0.427** | 0.432/0.427 | 0.456/0.452 | 0.454/0.448 | 0.469/0.455 |
| ETTh2 | 0.371/0.399 | 0.383/0.404 | 0.559/0.515 | 0.383/0.407 | 0.387/0.407 |
| Weather | 0.243/0.270 | 0.254/0.279 | 0.265/0.317 | 0.258/0.278 | 0.259/0.281 |
| ECL | **0.168/0.259** | 0.170/0.260 | 0.212/0.300 | 0.178/0.270 | 0.205/0.290 |
| Traffic | **0.448/0.261** | 0.485/0.313 | 0.625/0.383 | 0.428/0.282 | 0.481/0.304 |

FreqCycle achieves first place on 10 out of 14 metrics and second place on 2.

### Ablation Study

| Configuration | ETTh1 MSE | ETTh2 MSE | Traffic MAE | Weather MSE |
|------|-----------|-----------|-------------|-------------|
| FreqCycle (full) | 0.369 | 0.282 | 0.262 | 0.159 |
| w/o FECF | 0.375 (+1.65%) | 0.292 (+3.26%) | 0.318 (+21.30%) | 0.179 (+12.59%) |
| SFPL → MLP | 0.379 (+2.66%) | 0.302 (+6.91%) | 0.262 (+0.04%) | 0.161 (+1.51%) |

| Frequency-domain method comparison | ETTh2 MSE | ECL MSE | Traffic MSE |
|----------------|-----------|---------|-------------|
| FreqCycle (SFPL) | **0.282** | **0.139** | **0.438** |
| SFPL → LPF (FITS) | 0.294 (+4.00%) | 0.143 (+3.39%) | 0.449 (+2.63%) |
| SFPL → PaiFilter | 0.293 (+3.68%) | 0.143 (+2.96%) | 0.445 (+1.81%) |

### MFreqCycle Long Lookback Window Experiments

| Dataset | L | FreqCycle MSE | CycleNet MSE | FITS MSE |
|--------|---|--------------|-------------|----------|
| ETTh2 | 96 | 0.371 | 0.383 | 0.383 |
| ETTh2 | 168 | **0.241** | 0.388 | 0.333 |
| ETTm2 | 96 | 0.263 | 0.267 | 0.286 |
| ETTm2 | 672 | **0.167** | 0.262 | 0.250 |

MFreqCycle achieves substantial performance gains with long lookback windows (ETTm2: 0.263 → 0.167).

### Key Findings

1. **Complementarity of FECF and SFPL**: FECF yields the largest gains on datasets with strong periodicity (Traffic, Weather), while SFPL excels on datasets with strong non-stationarity (ETTh series).
2. **SFPL outperforms traditional frequency-domain filtering**: Compared to low-pass filtering (FITS) and PaiFilter, SFPL more effectively extracts mid-to-high frequency information through segmented enhancement.
3. **Efficiency advantage**: FreqCycle achieves the optimal balance across forecasting performance, peak memory consumption, and training speed (Figure 4).
4. **Visualization validation**: Spectral analysis clearly demonstrates that SFPL enhances the energy contribution of mid-to-high frequency components while preserving the structure of low-frequency components (Figure 5).

## Highlights & Insights

1. **Explicit vs. implicit periodicity modeling**: Rather than implicitly capturing periodicity through complex architectures, the framework directly learns via a parameterized periodic basis — a simple yet effective approach.
2. **Importance of mid-to-high frequency components**: This work is the first to systematically demonstrate and address the inherent limitations of MLP-based methods in modeling mid-to-high frequency components.
3. **Segmentation strategy inspired by STFT**: Incorporating the Short-Time Fourier Transform concept from signal processing into deep learning provides a solid theoretical foundation.
4. **Practical value of multi-scale design**: The substantial gains of MFreqCycle under long lookback windows validate the value of decoupling nested periodic features.

## Limitations & Future Work

1. **Base period W requires manual specification**: Relying on prior knowledge of data properties (e.g., sampling frequency), automatically determining the optimal combination of periods remains an open problem.
2. **MFreqCycle validated only up to weekly periodicity**: Longer periods such as annual cycles have not been verified due to dataset limitations.
3. **Channel-independent assumption**: Inter-variable correlations are not explicitly modeled.
4. Integration with Transformer architectures to leverage attention mechanisms for cross-channel dependency modeling is a promising direction.
5. The relationship between the choice of sub-segment count $s$ in the segmentation strategy and data characteristics warrants further investigation.

## Related Work & Insights

- **CycleNet** (Lin et al., 2024a): The predecessor of FECF in this work; explicitly models periodic patterns but lacks frequency-domain enhancement.
- **FITS** (Xu et al., 2024): A frequency-domain low-pass filtering method that focuses exclusively on low-frequency components.
- **FilterNet** (Yi et al., 2024): A learnable filter-based method; SFPL in this work further improves upon this approach.
- Insight: In time series forecasting, time-domain and frequency-domain methods each have distinct strengths; combining the two (FECF for low-frequency and SFPL for mid-to-high frequency) achieves complementary coverage.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The mid-to-high frequency enhancement in SFPL is a meaningful contribution, though the overall framework is an incremental combination of existing ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 7 datasets, multi-level ablations, efficiency analysis, and visualization validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with sufficient theoretical motivation, though some mathematical derivations could be more concise.
- **Value**: ⭐⭐⭐⭐ — Achieves a good balance between performance and efficiency, with practical contributions to MLP-based TSF methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[ICML 2026\] Generalizing Multi-scale Time-Series Modeling with a Single Operator](../../ICML2026/time_series/generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[AAAI 2026\] A Theoretical Analysis of Detecting Large Model-Generated Time Series](a_theoretical_analysis_of_detecting_large_model-generated_time_series.md)

</div>

<!-- RELATED:END -->
