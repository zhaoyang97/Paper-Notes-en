---
title: >-
  [Paper Note] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting
description: >-
  [AAAI 2026][Time Series][Extreme event forecasting] This paper proposes M2FMoE, a framework that models both regular and extreme temporal patterns via frequency-domain Mixture-of-Experts from dual Fourier and wavelet perspectives. It incorporates a cross-view shared frequency-band splitter to align semantic correspondence across domains, multi-resolution adaptive fusion to capture multi-scale information, and temporal gated integration to combine short- and long-term features…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "Extreme event forecasting"
  - "frequency-domain modeling"
  - "Mixture-of-Experts"
  - "wavelet transform"
  - "Fourier transform"
  - "multi-resolution fusion"
  - "hydrological forecasting"
date: 2026-05-08
content_hash: dad3999eaa5f3f21
---

# M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting

**Conference**: AAAI 2026
**arXiv**: [2601.08631](https://arxiv.org/abs/2601.08631)  
**Code**: [https://github.com/Yaohui-Huang/M2FMoE](https://github.com/Yaohui-Huang/M2FMoE)  
**Area**: Time Series Forecasting
**Keywords**: Extreme event forecasting, frequency-domain modeling, Mixture-of-Experts, wavelet transform, Fourier transform, multi-resolution fusion, hydrological forecasting

## TL;DR
This paper proposes M2FMoE, a framework that models both regular and extreme temporal patterns via frequency-domain Mixture-of-Experts from dual Fourier and wavelet perspectives. It incorporates a cross-view shared frequency-band splitter to align semantic correspondence across domains, multi-resolution adaptive fusion to capture multi-scale information, and temporal gated integration to combine short- and long-term features. On five hydrological extreme event datasets, M2FMoE surpasses all state-of-the-art methods — including label-supervised approaches — without requiring any extreme event labels, achieving an average RMSE improvement of 22.3%.

## Background & Motivation

**Background**: Time series forecasting is critical in energy, transportation, and environmental monitoring. Extreme events in hydrological forecasting — such as rainstorms, floods, and sudden water-level surges — are particularly difficult to predict due to their rarity, abruptness, and high variance.

**Limitations of Prior Work**:
   - Mainstream deep learning models (Transformers, MLPs, etc.) focus on dominant regular patterns (periodicities, smooth trends) and fail to capture the irregular high-frequency abrupt changes characteristic of extreme events.
   - Frequency-domain methods each have inherent drawbacks: FFT provides global frequency information but lacks temporal localization, while wavelets offer time-frequency localization but have insufficient resolution at low frequencies.
   - Existing extreme-adaptive methods (DAN, MCANN, etc.) rely on extreme event labels as auxiliary supervision, limiting their generalizability.
   - Dual-view strategies introduce cross-view spectral misalignment: FFT's uniform frequency axis and CWT's nonlinear scale axis cause the same signal to occupy inconsistent positions across the two domains.

**Key Challenge**: Extreme and regular events exhibit markedly different spectral characteristics — extreme events display broad-band multi-peak slow decay, whereas regular events concentrate energy in narrow low-frequency bands. Adaptive focus on different frequency bands is therefore required, and no single frequency-domain view can simultaneously capture both global and local information.

**Goal**: Jointly capture regular and extreme temporal patterns through multi-view, multi-resolution frequency-domain modeling, without relying on any extreme event labels.

**Key Insight**: Leverage the expert specialization capability of MoE to assign different frequency bands to different experts, and employ a shared frequency-band splitter to resolve the cross-view alignment problem.

**Core Idea**: FFT experts capture global periodicity + wavelet experts capture local abrupt changes → shared frequency-band splitter aligns the two domains → multi-resolution fusion from coarse to fine → gated integration of short- and long-term representations.

## Method

### Overall Architecture
M2FMoE comprises three major modules:
1. **Multi-view Frequency Mixture-of-Experts (MFMoE)**: an FFT branch and a wavelet branch, each containing $E$ frequency-band experts.
2. **Multi-resolution Adaptive Fusion (MAF)**: aggregates features from multiple temporal resolutions.
3. **Temporal Gated Integration (TGI)**: adaptively fuses short-term predictions with long-term historical representations.

The input signal is first decomposed by **hierarchical temporal segmentation** into a recent segment $\mathbf{X}_r$ and the full history $\mathbf{X}$. The recent segment is passed through multi-resolution smoothing convolutions to generate difference sequences $\Delta \mathbf{X}_r^{(k)}$ at varying granularities, which are then fed into MFMoE.

### Key Design 1: Cross-View Shared Frequency-Band Splitter (CSS)
- Learns shared frequency boundaries $\{\beta_1, \ldots, \beta_{E-1}\}$ that partition the frequency range $[0,1]$ into $E$ bands.
- **FFT view**: boundaries are directly scaled to frequency indices $\{\tilde{\beta}_e\}$.
- **Wavelet view**: frequency boundaries are nonlinearly mapped to wavelet scale indices $\{\ddot{\beta}_e\}$ via the inverse mapping from Theorem 1 ($a = \gamma/f$).
- **Theoretical guarantee**: Theorem 1 proves that the frequency-scale mapping preserves signal energy conservation.
- **Core value**: ensures that experts across both views operate on semantically consistent frequency bands, eliminating cross-view misalignment.

### Key Design 2: Dual-View Frequency Expert Branches
**FFT Branch** (Fig. 2c):
- Applies per-channel FFT to the difference sequences; binary masks $\tilde{\mathbb{I}}_e$ isolate each expert's frequency band.
- Lightweight routing network: channel-averaged amplitude spectrum → two-layer Linear+ReLU+Softmax → routing weights $\boldsymbol{\alpha}$.
- Weighted aggregation of expert frequency bands → IFFT → linear projection, producing output $\tilde{\mathbf{V}} \in \mathbb{R}^{T_p \times C}$.

**Wavelet Branch** (Fig. 2d):
- Applies CWT with a complex Gaussian wavelet to obtain the power spectrum $\mathcal{P} = |\mathcal{W}(a,b)|^2$.
- Scale masks $\ddot{\mathbb{I}}_e$ isolate each expert's scale range.
- Each expert processes its input with a two-layer convolutional block: Conv → ReLU → Dropout → Conv.
- An analogous routing mechanism produces gating weights $\boldsymbol{\eta}$; outputs are weighted-aggregated, flattened, and projected through two linear layers.

**Complementarity**: FFT experts focus on low-frequency global trends, wavelet experts on high-frequency local abrupt changes; the routing mechanism dynamically adjusts weights according to the spectral characteristics of the input.

### Key Design 3: Multi-Resolution Adaptive Fusion (MAF) + Temporal Gated Integration (TGI)
**MAF**:
- **Multi-view fusion**: FFT and wavelet branch outputs are concatenated along the channel dimension, augmented with temporal encodings, and fused via BN + Linear.
- **Multi-resolution fusion**: fused representations at different resolutions are each passed through a dedicated linear transformation and summed: $\mathbf{H}_r = \sum_{i=1}^{R} \text{Linear}_i(\mathbf{H}_u^{(i)})$.
- The last observed value is added back to recover the original value range.

**TGI**:
- The historical input $\mathbf{X}$ is linearly projected to obtain $\mathbf{H}_h$.
- Gating coefficient: $\mathbf{G} = \sigma(\text{Linear}([\mathbf{H}_r; \mathbf{H}_h]))$.
- Final output: $\hat{\mathbf{X}} = \mathbf{G} \odot \mathbf{H}_r + (1-\mathbf{G}) \odot \mathbf{H}_h$.

### Loss & Training
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{div}} + \mu \mathcal{L}_{\text{cons}}$$

- $\mathcal{L}_{\text{pred}}$: MSE prediction loss.
- $\mathcal{L}_{\text{div}}$: expert diversity loss, encouraging differentiated outputs across experts (standard deviation penalty).
- $\mathcal{L}_{\text{cons}}$: cross-view consistency loss, encouraging cosine similarity between expert outputs for the same frequency band across the FFT and wavelet views.

## Key Experimental Results

### Main Results (5 reservoirs, forecasting horizons of 8h and 72h)

| Dataset | H(h) | M2FMoE RMSE | CATS RMSE | FreqMoE RMSE | iTrans RMSE | MCANN RMSE (w/ labels) |
|--------|------|-------------|-----------|-------------|-------------|---------------------|
| Almaden | 8 | **7.99** | 16.09 | 14.73 | 32.13 | 8.45 |
| Coyote | 8 | **48.80** | 110.85 | 593.14 | 372.52 | 86.83 |
| Lexington | 8 | **251.96** | 618.99 | 387.00 | 690.43 | 253.0 |
| Stevens Cr. | 8 | **10.56** | 18.50 | 80.94 | 48.88 | 12.13 |
| Vasona | 8 | **5.13** | 6.91 | 14.32 | 12.18 | 5.35 |
| Coyote | 72 | **449.94** | 509.08 | 855.10 | 673.85 | 559.75 |
| Lexington | 72 | **772.84** | 906.53 | 1003.82 | 960.65 | 778.02 |

- Average rank 1.4 (vs. best label-free baseline CATS at 3.7; label-supervised MCANN at 1.7).
- Average RMSE improvement of **22.30%** over the best label-free baseline, with a maximum gain of 52.86% (Coyote, 8h).
- Average RMSE improvement of **9.19%** over label-supervised MCANN, with a maximum gain of 43.8%.

### Ablation Study (forecasting horizon 72h)

| Ablation Variant | Almaden | Coyote | Lexington | Stevens Cr. | Vasona |
|---------|---------|--------|-----------|-------------|--------|
| Full M2FMoE | **54.12** | **449.94** | **772.84** | **76.94** | **19.57** |
| w/o wavelet branch | 57.70 | 555.64 | 827.39 | 87.19 | 19.84 |
| w/o FFT branch | 59.04 | 558.26 | 870.74 | 85.02 | 19.81 |
| w/o multi-resolution | 59.48 | 483.22 | 855.24 | 85.00 | 19.51 |
| w/o CSS | 55.58 | 541.59 | 916.93 | 86.00 | 20.00 |
| w/o alignment | 59.12 | 516.73 | 872.03 | 85.80 | 19.28 |
| w/o dual-view | 56.27 | 453.46 | 789.04 | 79.06 | 19.38 |

### Key Findings
1. **Dual-view complementarity is effective**: removing either branch causes a clear performance degradation; removing the FFT branch has a slightly larger impact due to the loss of global trend modeling.
2. **Cross-view alignment via CSS is critical**: removing CSS increases RMSE by 18.6% on Lexington, demonstrating that semantically consistent frequency-band splitting is essential for dual-view collaboration.
3. **Wavelet experts are more strongly activated during extreme events**: t-SNE visualizations show that the wavelet view is more sensitive to high-frequency local abrupt changes, while the FFT view better handles stable periodic patterns.
4. **Optimal number of experts is 3–4**: too many experts introduce noise and overfitting.
5. **Recent segment length has a sweet spot**: too short is informationally insufficient; too long introduces noise; omitting the recent segment entirely causes a significant performance drop.
6. M2FMoE remains competitive with PatchTST/TimesNet on standard datasets such as ETTh, with advantages more pronounced at shorter horizons.

## Highlights & Insights
- **Surpasses label-supervised methods without any extreme event labels**: fully end-to-end, yielding stronger generalizability.
- **Rigorous cross-domain alignment theory**: Theorem 1 provides an energy-conservation proof for the frequency-scale mapping, upon which CSS learns shared boundaries in a principled manner.
- **Spectral heterogeneity as a novel lens for extreme events**: the paper presents the first systematic analysis of spectral differences between extreme and regular events (Fig. 1), providing intuitive motivation for frequency-domain modeling.
- **Nested MoE + multi-resolution design**: frequency-band-level expert specialization → view-level fusion → resolution-level aggregation → temporal gating, forming a hierarchical and functionally well-delineated architecture.

## Limitations & Future Work
1. Validation is limited to hydrological datasets; generalizability to other extreme event domains (e.g., finance, transportation) has yet to be demonstrated.
2. CWT incurs higher computational cost than FFT (via the PyWavelets library), which may become a bottleneck for very long sequences.
3. Performance on long-horizon benchmarks (e.g., ETTh1/h2, 96→96) is inferior to PatchTST/TimesNet, suggesting the method is better suited to short-to-medium-term extreme event scenarios.
4. Hyperparameters such as the number of experts and number of resolutions require per-dataset tuning, with limited automation.
5. The loss weights $\lambda$ and $\mu$ for the diversity and consistency terms must be set manually.

## Related Work & Insights
- **Frequency-domain time series forecasting**: FEDformer (frequency-enhanced decomposition), FreqMoE (frequency-decomposition MoE), U-Mixer (UNet mixer).
- **Extreme event methods**: NEC+/VIE/SADI/SEED (multi-stage learning), DAN (extreme-value labels), MCANN (cluster attention), EPL/EVL/GEVL (extreme value theory losses).
- **MoE time series models**: Time-MoE (large-scale foundation model), FreqMoE (frequency decomposition).
- **Multi-resolution / multi-scale**: TimeMixer (multi-scale mixing), TimeMixer++ (enhanced variant).
- **Attention-based methods**: CATS (cross-attention), iTransformer (inverted Transformer).

## Rating
⭐⭐⭐⭐ — Framing extreme event forecasting through the lens of spectral heterogeneity is highly novel. CSS provides a rigorous theoretical foundation for cross-domain alignment, and surpassing label-supervised methods without labels is a compelling result. The modular design is hierarchically clear with well-defined functional roles. Limitations include a narrow validation scope (hydrological data only) and moderate performance on long-horizon forecasting. Overall, this represents a solid and meaningful advance in extreme-event time series forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[ICCV 2025\] VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting](../../ICCV2025/time_series/va-moe_variables-adaptive_mixture_of_experts_for_incremental_weather_forecasting.md)
- [\[AAAI 2026\] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)
- [\[ICLR 2026\] Improving Extreme Wind Prediction with Frequency-Informed Learning](../../ICLR2026/time_series/improving_extreme_wind_prediction_with_frequency-informed_learning.md)

</div>

<!-- RELATED:END -->
