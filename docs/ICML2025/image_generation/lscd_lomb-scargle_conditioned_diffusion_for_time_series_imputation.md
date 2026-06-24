---
title: >-
  [Paper Note] LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation
description: >-
  [ICML 2025][Image Generation][Time series imputation] This paper proposes LSCD, which integrates a differentiable Lomb-Scargle periodogram layer into a score-based diffusion model for time series imputation. Through frequency-domain conditioning information and a spectral consistency loss, the approach simultaneously improves time-domain imputation accuracy and frequency-domain recovery consistency under high missing rates.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Time series imputation"
  - "Lomb-Scargle periodogram"
  - "frequency-domain conditional generation"
  - "irregular sampling"
  - "differentiable spectral layer"
date: 2026-05-08
content_hash: ce649e748527a9dc
---

# LSCD: Lomb-Scargle Conditioned Diffusion for Time Series Imputation

**Conference**: ICML 2025  
**arXiv**: [2506.17039](https://arxiv.org/abs/2506.17039)  
**Code**: None  
**Area**: Diffusion Models / Time Series  
**Keywords**: Time series imputation, Lomb-Scargle periodogram, frequency-domain conditional generation, irregular sampling, differentiable spectral layer

## TL;DR
This paper proposes LSCD, which integrates a differentiable Lomb-Scargle periodogram layer into a score-based diffusion model for time series imputation. Through frequency-domain conditioning information and a spectral consistency loss, the approach simultaneously improves time-domain imputation accuracy and frequency-domain recovery consistency under high missing rates.

## Background & Motivation
**Background**: Time series missing value imputation methods mostly operate in the time domain (BRITS, SAITS, CSDI). Some methods utilize FFT to extract frequency-domain features (TimesNet), but FFT requires uniform sampling.

**Limitations of Prior Work**: When data is missing, FFT requires prior interpolation or zero padding, which yields severe spectral distortion under high missing rates. Existing diffusion imputation methods (such as CSDI) also operate solely in the time domain, neglecting the frequency structure of signals.

**Key Challenge**: Spectral analysis of irregularly sampled or missing data remains a fundamental challenge, as the uniform sampling assumption of FFT does not align with practical scenarios.

**Goal**: How to provide robust spectral estimation for irregularly sampled data and integrate it into the diffusion generation process?

**Key Insight**: The Lomb-Scargle periodogram naturally supports spectral analysis of irregularly sampled data, and can be differentiated to be incorporated into end-to-end learning.

**Core Idea**: Replace FFT with a differentiable Lomb-Scargle layer to provide frequency-domain conditioning information for the conditional diffusion model without requiring interpolation.

## Method

### Overall Architecture
Building on the conditional diffusion framework of CSDI, LSCD: (1) utilizes a Lomb-Scargle layer to compute the spectrum from observed data, serving as an additional conditional input for the denoising network; (2) encodes the spectrum into a conditional representation via an attention encoder; (3) introduces a spectral consistency loss during the late stage of training to align the imputed results with the observed spectrum.

### Key Designs

1. **Differentiable Lomb-Scargle Layer**:

    - Function: Directly computes power spectral density from irregularly sampled/missing data.
    - Mechanism: $P(\omega) = \frac{(\sum_i [x_{s_i} - \bar{x}]\cos[\omega\phi_i])^2}{\sum_i \cos^2[\omega\phi_i]} + \frac{(\sum_i [x_{s_i} - \bar{x}]\sin[\omega\phi_i])^2}{\sum_i \sin^2[\omega\phi_i]}$, where $\phi_i = s_i - \tau$ ensures time-shift invariance.
    - Design Motivation: No interpolation is required; it directly fits sinusoidal functions to the observed points, making it applicable to arbitrary missingness patterns.

2. **Attention-based Spectral Encoder $\mathcal{E}_{\text{spec}}$**:

    - Function: Encodes the LS spectrum into a conditional representation to inject into the denoising process.
    - Mechanism: A two-layer multi-head self-attention mechanism captures dependencies across frequencies and features, generating $\mathbf{z}_S$ as an auxiliary condition for each denoising step.
    - Design Motivation: The complete spectrum contains rich information; the model needs to learn which frequency components are most relevant to the imputation task.

3. **Spectral Consistency Loss $\mathcal{L}_{\text{SCons}}$**:

    - Function: Ensures that the spectrum of the imputed results remains consistent with the observed spectrum in the late stages of training.
    - Mechanism: $\mathcal{L}_{\text{SCons}} = \|\mathcal{LS}(\mathbf{x}_0^{co}) - \mathcal{LS}(\hat{\mathbf{x}}_0^{co})\|_2^2$, comparing the original spectrum of the observed part with the reconstructed spectrum.
    - Design Motivation: Time-domain losses cannot guarantee frequency-domain consistency; this loss ensures that the frequency structure is preserved.

### Loss & Training
- Primary Loss: Standard diffusion denoising loss $\mathcal{L} = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2]$.
- Spectral consistency loss $\mathcal{L}_{\text{SCons}}$ is added in the late stage of training, which requires obtaining $\hat{\mathbf{x}}_0$ through a complete reverse diffusion process.
- False Alarm Probability (FAP) is utilized to filter out unreliable frequency components.

## Key Experimental Results

### Main Results

| Dataset | Missing Rate | Metric | CSDI | SAITS | **LSCD** |
|--------|--------|------|------|-------|----------|
| Sines (point) | 10% | MAE ↓ | 1.336 | 0.885 | **0.765** |
| Sines (point) | 50% | MAE ↓ | 1.359 | 1.041 | **0.975** |
| Sines (point) | 90% | MAE ↓ | 1.361 | 1.292 | **1.271** |
| Sines (point) | 10% | S-MAE ↓ | 0.008 | 0.043 | **0.003** |
| Sines (point) | 90% | S-MAE ↓ | 0.044 | 0.375 | **0.036** |

### Ablation Study

| Component | Effect | Description |
|------|------|------|
| No LS condition (baseline CSDI) | Highest S-MAE | Lack of frequency-domain information |
| + LS condition | Significant decrease in S-MAE | Frequency-domain guidance is highly critical |
| + Spectral Encoder | Further decrease in MAE | Learning frequency weights |
| + Spectral Consistency Loss | Best on both metrics | Frequency-domain alignment |

### Key Findings
- The LS condition contributes most significantly to spectral recovery (S-MAE) (CSDI 0.044 vs. LSCD 0.036 under 90% missingness).
- Under a high missing rate (90%), the superiority of LSCD is more pronounced because FFT fails completely under high missingness.
- LSCD shows larger improvements over CSDI under sequence-missing and block-missing scenarios.

## Highlights & Insights
- **Generality of Differentiable LS Layer**: It is applicable not only to imputation but can also be integrated into any deep learning pipeline that handles irregular frequency-domain information.
- **Systematic Comparison of FFT vs. LS**: It clearly demonstrates the spectral distortion issues of FFT under missing data conditions.
- **Transferable Concepts**: The concept of LS conditioning can be transferred to other tasks such as time series forecasting and anomaly detection.

## Related Work & Insights
- **vs. CSDI**: CSDI is a representative method for pure time-domain conditional diffusion imputation. LSCD adds frequency-domain conditioning and consistency loss to its architecture.
- **vs. TimesNet**: TimesNet employs FFT to extract periodic features, but requires prior interpolation of missing values. LSCD's LS layer directly handles missing values.
- **vs. BRITS/SAITS**: These deterministic approaches fail to provide uncertainty estimation, whereas LSCD, as a probabilistic model, can provide distributions through multiple samplings.
- The differentiable LS layer can be used independently of LSCD, offering integration into any deep learning pipeline requiring frequency-domain analysis.

## Limitations & Future Work
- The time complexity of the LS layer is $O(LJ)$ ($L$ observed points $\times$ $J$ frequencies), which is slower than FFT's $O(L\log L)$.
- Evaluation is limited to synthetic sine waves and two real-world datasets, lacking validation on large-scale time series benchmarks (e.g., ETTh, Weather).
- The spectral consistency loss requires a full reverse diffusion process, yielding higher training costs.
- The effectiveness of spectral conditioning on non-periodic signals (e.g., trend-dominated time series) remains unverified.
- Modeling of spectral correlation across multiple variables has not been fully exploited.

## Rating
- Novelty: ⭐⭐⭐⭐ Integrating Lomb-Scargle into deep learning is a meaningful first step.
- Experimental Thoroughness: ⭐⭐⭐ The datasets utilized are limited, lacking large-scale validation.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations and comprehensive background context.
- Value: ⭐⭐⭐⭐ Provides a feasible solution for frequency-domain modeling of irregular time series.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modeling](bridge_bootstrapping_text_to_control_time-series_generation_via_multi-agent_iter.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](../../NeurIPS2025/image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [\[NeurIPS 2025\] TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising](../../NeurIPS2025/image_generation/tidmad_time_series_dataset_for_discovering_dark_matter_with_ai_denoising.md)
- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [\[AAAI 2026\] SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting](../../AAAI2026/image_generation/simdiff_simpler_yet_better_diffusion_model_for_time_series_point_forecasting.md)

</div>

<!-- RELATED:END -->
