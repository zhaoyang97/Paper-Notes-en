---
title: >-
  [Paper Note] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing
description: >-
  [AAAI 2026][Time Series][Non-stationary time series] This paper proposes DTAF, a dual-branch framework that extracts and removes heterogeneous non-stationary patterns via a non-stationary MoE filter in the temporal domai…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Non-stationary time series"
  - "mixture of experts"
  - "frequency differencing"
  - "dual-branch modeling"
  - "time series forecasting"
date: 2026-05-08
content_hash: 2327115f82aa2bc3
---

# Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing

**Conference**: AAAI 2026
**arXiv**: [2511.08229](https://arxiv.org/abs/2511.08229)  
**Code**: [https://github.com/decisionintelligence/DTAF](https://github.com/decisionintelligence/DTAF)  
**Area**: Time Series
**Keywords**: Non-stationary time series, mixture of experts, frequency differencing, dual-branch modeling, time series forecasting

## TL;DR

This paper proposes DTAF, a dual-branch framework that extracts and removes heterogeneous non-stationary patterns via a non-stationary MoE filter in the temporal domain, tracks frequency drift via spectral differencing in the frequency domain, and fuses complementary information from both domains through dual-branch attention for robust non-stationary time series forecasting.

## Background & Motivation

### Problem Background

Time series forecasting is critical in energy, finance, transportation, and cloud computing. Real-world time series commonly exhibit non-stationarity in **both the temporal and frequency domains**:
- **Temporal non-stationarity**: Large local distribution shifts across different time periods (e.g., changes in mean and variance), which interfere with long-term dependency modeling.
- **Frequency non-stationarity**: Frequency components vary over time (e.g., daily and seasonal cycles in electricity load shift due to user behavior and climate factors).

### Two Core Challenges

**Challenge 1: Extracting and separating heterogeneous non-stationary patterns**

Time series inherently contain both stationary and non-stationary components. Directly modeling long-term dependencies on raw sequences is severely disrupted by non-stationary dynamics. Although stationary components can be explicitly modeled and non-stationary effects isolated, non-stationary patterns are highly complex and heterogeneous, making it infeasible for **a single architecture to comprehensively model** all such patterns.

**Challenge 2: Dynamically modeling frequency drift**

Existing frequency analysis methods (e.g., FFT) assume stationarity within the observation window and produce global frequency representations, **failing to capture transient frequency shifts and local patterns**. For example, in electricity load forecasting, daily and seasonal cycles continuously change due to user behavior and environmental factors.

### Core Idea

Non-stationarity is addressed jointly from temporal and frequency perspectives: the temporal domain employs a MoE to learn and filter diverse non-stationary patterns, while the frequency domain uses differencing to track spectral changes.

## Method

### Overall Architecture

DTAF consists of the following modules:
1. **Instance Norm**: Mitigates distribution shift between training and inference.
2. **Patching & Embedding**: Segments long sequences into patches and embeds them into high-dimensional space.
3. **Temporal Stabilizing Fusion (TFS)**: Temporal non-stationarity handling + long-term dependency modeling.
4. **Frequency Wave Modeling (FWM)**: Frequency-domain non-stationarity modeling.
5. **Dual-branch Attention**: Fuses temporal and frequency features.
6. **Predictor (FC)**: Final prediction.

### Key Designs

#### 1. **Temporal Stabilizing Fusion (TFS)**

TFS consists of two sub-modules: the Non-stationary MoE Filter and Temporal Fusion.

**Non-stationary MoE Filter**:

Core objective: learn and **remove** non-stationary patterns from the input to obtain an approximately stationary representation.

- Composed of multiple experts (each implemented as an independent MLP), with each expert specializing in extracting a specific type of non-stationary pattern.
- A routing network based on a KAN linear layer + Softmax dynamically assigns expert weights per patch.
- Extracted non-stationary patterns are subtracted from the original patches:

$$\mathbf{X}_{\mathrm{stable}}^i = \mathbf{X}_{\mathrm{patch}}^i - \mathbf{X}_{\mathrm{patterns}}^i$$

- Key constraint: a **KL divergence loss** ensures that experts genuinely learn non-stationary patterns (i.e., after removal, the distributions of all patches should converge):

$$\mathcal{L}_{\mathrm{stable}} = \alpha \sum_{i=1}^{N} \sum_{j=1}^{N} \mathrm{KL}(\mathbf{X}_{\mathrm{stable}}^i, \mathbf{X}_{\mathrm{stable}}^j) / N^2$$

**Design Motivation**: Different patches may contain different types of non-stationary patterns (trend changes, abrupt shifts, variance drift, etc.). A single model cannot cover all of them; hence, multiple experts divide and conquer.

**Temporal Fusion**:

After obtaining a near-stationary representation, long-term dependencies are modeled as follows:

1. **Feature extraction**: Each patch undergoes temporal decomposition (trend + seasonality), with results linearly transformed and fused: $\mathbf{X}_h^i = \mathbf{W}_i^t \cdot \mathbf{X}_t^i + \mathbf{W}_i^s \cdot \mathbf{X}_s^i$
2. **Historical weight generation**: A linear layer computes fusion weights between the current patch and each historical patch (causal masking prevents future information leakage).
3. **Weighted aggregation**: $\mathbf{X}_{\mathrm{history}}^i = \mathrm{MLP}(\sum_{n=1}^{i-1} \mathbf{Weight}_n^i \cdot \mathbf{X}_{\mathrm{stable}}^n)$
4. **Gating mechanism**: Dynamically regulates the contribution of the current patch: $\mathbf{X}_{\mathrm{current}}^i = \mathrm{Gate}(\mathbf{X}_{\mathrm{patch}}^i) \cdot \mathbf{X}_{\mathrm{patch}}^i$
5. **Fusion**: $\mathbf{H}_t^i = \mathbf{X}_{\mathrm{current}}^i + \mathbf{X}_{\mathrm{history}}^i$

#### 2. **Frequency Wave Modeling (FWM)**

Core innovation: introducing **differencing operations in the frequency domain** to track spectral changes over time.

Steps:
1. Apply rFFT to the temporal representation of each patch to obtain its spectrum: $\mathbf{Freq}^i = \mathrm{rFFT}(\mathbf{H}_t^i)$
2. Compute spectral differences between adjacent patches: $\mathbf{Wave}^i = \mathbf{Freq}^i - \mathbf{Freq}^{i-1}$
3. Select the Top-K frequency components with the most significant changes: $\mathbf{Picks}^i = \mathrm{TopK}(\mathbf{Wave}^i)$
4. Zero out unselected frequency components.
5. Apply inverse FFT to transform back to the temporal domain.

**Design Motivation**: Conventional FFT yields a global frequency representation that cannot distinguish between "stable periodic patterns" and "frequency components currently undergoing change." The differencing operation directly highlights the most dynamic frequencies, directing the model's attention to non-stationary frequency-domain dynamics.

#### 3. **Dual-branch Attention**

Temporal features from TFS and frequency features from FWM are each processed by independent self-attention modules and then concatenated:

$$\mathbf{H}_{\mathrm{fusion}} = \mathrm{Concat}(\mathbf{Atten}_t, \mathbf{Atten}_f) \in \mathbb{R}^{2N \times d}$$

The final prediction is produced by an FC layer.

### Loss & Training

The total loss consists of three components:

$$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \alpha \cdot \mathcal{L}_{\mathrm{stable}} + \beta \cdot \mathcal{L}_{\mathrm{robust}}$$

- $\mathcal{L}_{\mathrm{task}}$: L1 loss.
- $\mathcal{L}_{\mathrm{stable}}$: KL divergence constraint on the non-stationary MoE filter.
- $\mathcal{L}_{\mathrm{robust}}$: R-Drop loss for improved robustness.

## Key Experimental Results

### Main Results

Multivariate forecasting results on 11 real-world datasets (MSE / MAE, lower is better):

| Dataset | DTAF (MSE/MAE) | Amplifier | iTransformer | PatchTST | Stationary | Gain |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| ILI | **1.688/0.801** | 1.819/0.888 | 1.857/0.892 | 1.902/0.879 | 2.389/1.027 | vs Stationary: MSE −29.4% |
| Covid-19 | **1.351/0.040** | 5.578/0.112 | 1.488/0.049 | 1.697/0.056 | 2.658/0.078 | vs Stationary: MSE −49.2% |
| NN5 | **0.643/0.538** | 1.794/1.018 | 0.660/0.550 | 0.698/0.582 | 1.295/0.915 | vs Stationary: MSE −50.3% |
| ETTh | **0.369/0.398** | 0.385/0.416 | 0.404/0.425 | 0.384/0.415 | 0.521/0.493 | vs Stationary: MSE −29.2% |
| ETTm | **0.297/0.338** | 0.327/0.364 | 0.314/0.358 | 0.302/0.347 | 0.456/0.430 | vs Stationary: MSE −34.9% |
| Weather | **0.222/0.250** | 0.222/0.263 | 0.232/0.269 | 0.223/0.261 | 0.293/0.315 | vs Stationary: MSE −24.2% |
| Electricity | **0.160/0.248** | 0.174/0.267 | 0.163/0.258 | 0.171/0.270 | 0.194/0.295 | vs Stationary: MSE −17.5% |
| Traffic | **0.402/0.249** | 0.423/0.294 | 0.397/0.281 | 0.397/0.275 | 0.621/0.339 | vs Stationary: MSE −35.3% |

Among 16 evaluation metrics, DTAF ranks first on 15. Performance is especially strong on non-stationary datasets (Covid-19, NN5).

### Ablation Study

Ablation results on NN5 (non-stationary) and ETTh1 (standard):

| Configuration | NN5-24 MSE | NN5-24 MAE | ETTh1-96 MSE | ETTh1-96 MAE | Note |
|:---|:---:|:---:|:---:|:---:|:---|
| **DTAF** | **0.716** | **0.559** | **0.359** | **0.384** | Full model |
| w/o TFS | 0.723 | 0.567 | 0.367 | 0.391 | Remove temporal module |
| w/o FWM | 0.729 | 0.574 | 0.363 | 0.391 | Remove frequency module |
| w/ Cross Attention | 0.725 | 0.571 | 0.376 | 0.395 | Replace dual-branch with cross-attention |

Each module is indispensable: TFS impacts long-term dependency modeling, FWM affects periodic pattern tracking, and dual-branch attention outperforms cross-attention.

### Key Findings

1. **Largest gains on non-stationary datasets**: MSE is reduced by 49% on Covid-19 and 50% on NN5, validating the effectiveness of the proposed non-stationarity handling.
2. **Interpretability of expert weight distributions**: Different samples and patches exhibit distinct expert weight allocations; samples with similar patterns share similar weight distributions.
3. **MoE filter demonstrably stabilizes distributions**: After filtering, the distributions of different patches visibly converge.
4. **Top-K frequency selection is not biased toward low frequencies**: Post-differencing Top-K selections are diverse across patches and are not dominated by high-energy low-frequency components.
5. **Dual-branch outperforms single-branch**: Using only the temporal or only the frequency branch is inferior to joint modeling.

## Highlights & Insights

1. **Novelty of frequency-domain differencing**: Most existing methods apply differencing in the temporal domain (e.g., ARIMA); applying differencing in the frequency domain to track spectral changes is a novel and intuitively motivated idea.
2. **MoE for heterogeneous non-stationary patterns**: Assigning different experts to different types of non-stationary patterns is more flexible than uniform processing.
3. **Systematic dual-domain complementary modeling**: Rather than naively stacking temporal and frequency processing, each branch independently handles non-stationarity before fusion, yielding a logically coherent design.
4. **Elegant KL constraint design**: By minimizing the KL divergence between patches after removing non-stationary components, the model is implicitly constrained to ensure experts genuinely capture non-stationary content.

## Limitations & Future Work

1. **A priori patch length selection**: The patching strategy requires a pre-specified length; adaptive patching schemes are not explored.
2. **Number of experts in the MoE**: The paper uses 4 experts without discussing the sensitivity to this hyperparameter or how to select it optimally.
3. **Frequency differencing only across adjacent patches**: Only spectral differences between adjacent time steps are computed; longer-span frequency changes are not considered.
4. **No comparison with recent LLM-based forecasting methods**: Foundation model approaches such as TimesFM and Chronos are not included as baselines.
5. **Computational efficiency**: The combination of dual-branch design, MoE, and FFT operations introduces additional computational overhead, which is not analyzed.

## Related Work & Insights

- **Non-stationary Transformer**: Addresses non-stationarity via input transformation, but does not model it in the frequency domain.
- **RevIN**: Uses instance normalization to handle distribution shift; DTAF builds on this by further modeling heterogeneous non-stationary patterns with a MoE.
- **TimesNet**: Leverages frequency-domain information for multi-period modeling, but assumes static frequency patterns.
- **PatchTST**: The source of the patching strategy; DTAF extends it by incorporating non-stationarity handling.
- Inspiration: The frequency differencing idea can be generalized to other tasks involving spectral changes, such as audio processing and signal analysis.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of frequency-domain differencing and a non-stationary MoE filter constitutes an innovative design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 11 datasets with comprehensive ablation studies and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and module design motivations are well articulated.
- **Value**: ⭐⭐⭐⭐ — Provides an effective solution to the important problem of non-stationary time series forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](../../ICML2026/time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[AAAI 2026\] IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)
- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](../../ICML2026/time_series/parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](../../NeurIPS2025/time_series/neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)

</div>

<!-- RELATED:END -->
