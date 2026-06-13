---
title: >-
  [Paper Note] Wavelet Canonical Coherence for Nonstationary Signals
description: >-
  [NeurIPS 2025][Time Series][Wavelet Analysis] This paper proposes WaveCanCoh, a framework that extends classical canonical coherence analysis to the wavelet domain. Built upon the multivariate locally stationary wavelet…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Wavelet Analysis"
  - "Canonical Coherence"
  - "Nonstationary Signals"
  - "Multivariate Time Series"
  - "Neuroscience"
date: 2026-05-08
content_hash: 8660b5baba6acffb
---

# Wavelet Canonical Coherence for Nonstationary Signals

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.14253](https://arxiv.org/abs/2505.14253)  
**Code**: [Available](https://github.com/mhaibo/WaveCanCoh)  
**Area**: Time Series Analysis / Signal Processing
**Keywords**: Wavelet Analysis, Canonical Coherence, Nonstationary Signals, Multivariate Time Series, Neuroscience

## TL;DR

This paper proposes WaveCanCoh, a framework that extends classical canonical coherence analysis to the wavelet domain. Built upon the multivariate locally stationary wavelet (MvLSW) model, it enables estimation of time-varying, scale-specific canonical coherence between two groups of nonstationary multivariate time series.

## Background & Motivation

Understanding the time-evolving dependence between two groups of multivariate signals is critical in fields such as neuroscience and finance—for example, analyzing functional interactions between electrode groups across different hippocampal subregions of rats during memory tasks.

**Core limitations of existing methods**:

**Stationarity assumption**: Classical canonical coherence analysis (Brillinger 1981) assumes weak stationarity, requiring statistical properties (mean, covariance, spectral characteristics) to remain constant over time. Real-world signals such as EEG/LFP are almost invariably nonstationary.

**Lack of temporal resolution**: Canonical coherence in the Fourier domain ignores temporal dynamics and cannot capture transient, frequency-specific interactions. Existing methods focus primarily on pairwise within-network coherence, lacking a global between-groups coherence measure.

**Absence of prior work**: To the authors' knowledge, no prior work has extended classical canonical coherence to the wavelet domain for measuring time-varying canonical coherence between two groups of multivariate nonstationary time series.

**Natural advantages of wavelet analysis**: Wavelet basis functions are simultaneously localized in time and frequency. Compactly supported wavelets can be compressed or stretched to adapt to the dynamic characteristics of signals, making them particularly suited for capturing transient properties of nonstationary signals.

## Method

### Overall Architecture

WaveCanCoh is built upon the multivariate locally stationary wavelet (MvLSW) process and involves three core steps:
1. **Concatenation**: Stack the two groups of time series $\mathbf{X}_t, \mathbf{Y}_t$ into $\mathbf{Z}_t$
2. **Spectral estimation**: Estimate the local wavelet spectrum (LWS) matrix
3. **Eigendecomposition**: Obtain time-varying canonical coherence and canonical direction vectors via eigenvalue decomposition

### Key Designs

1. **Local Wavelet Spectrum (LWS) Matrix**

   **Function**: Quantifies the local spectral contribution across channels at each scale $j$ and time location $u$.

   **Mechanism**: Under the MvLSW model, the process $\mathbf{X}_t$ is represented as $\mathbf{X}_t = \sum_j \sum_k \mathbf{V}_j(k/T) \psi_{j,k}(t) \mathbf{z}_{j,k}$, with the LWS matrix defined as $\mathbf{S}_j(u) = \mathbf{V}_j(u) \mathbf{V}_j^\top(u)$. This is extended to the cross-group LWS matrix:

    $\mathbf{S}_{j;\mathbf{ZZ}}(u) = \begin{bmatrix} \mathbf{S}_{j;\mathbf{XX}}(u) & \mathbf{S}_{j;\mathbf{XY}}(u) \\ \mathbf{S}_{j;\mathbf{YX}}(u) & \mathbf{S}_{j;\mathbf{YY}}(u) \end{bmatrix}$

   **Design Motivation**: The time-frequency localization of the wavelet basis allows the transfer matrix $\mathbf{V}_j(k/T)$ to capture the smoothly time-varying statistical properties of the process.

2. **Wavelet Canonical Coherence (WaveCanCoh) Definition**

   **Function**: Defines the maximum linear dependence between two groups of signals at each scale $j$ and time $u$.

   **Core formula**:

    $\boldsymbol{\rho}_{j;\mathbf{XY}}(u) = \max_{\mathbf{a}_j(u), \mathbf{b}_j(u)} \{\mathbf{a}_j^\top(u) \mathbf{S}_{j;\mathbf{XY}}(u) \mathbf{b}_j(u)\}^2$

   subject to $\mathbf{a}_j^\top \mathbf{S}_{j;\mathbf{XX}} \mathbf{a}_j = 1$ and $\mathbf{b}_j^\top \mathbf{S}_{j;\mathbf{YY}} \mathbf{b}_j = 1$.

   The solution is the largest eigenvalue $\Lambda_j^{(1)}(u)$ of the matrix $\mathbf{M}_{j;\mathbf{a}} = \mathbf{S}_{j,\mathbf{XX}}^{-1} \mathbf{S}_{j,\mathbf{XY}} \mathbf{S}_{j,\mathbf{YY}}^{-1} \mathbf{S}_{j,\mathbf{YX}}$.

3. **Causal Wavelet Canonical Coherence (Causal-WaveCanCoh)**

   **Function**: Incorporates time-lag relationships to capture potential causal effects.

   **Mechanism**: Defines the lagged joint process $\mathbf{Z}_t(h) = (\mathbf{X}_t^\top, \mathbf{Y}_{t+h}^\top)^\top$, constructs the lagged LWS matrix, and computes the causal canonical coherence accordingly.

4. **Estimation Procedure**

   The LWS matrix is estimated via the smoothed periodogram: $\hat{\mathbf{S}}_{j,k} = \sum_l A_{jl}^{-1} \tilde{\mathbf{I}}_{l,k}$, where $\tilde{\mathbf{I}}_{l,k}$ is the rectangle-window smoothed wavelet periodogram. Consistency is established under the regime $T, M \to \infty$ and $M/T \to 0$.

## Key Experimental Results

### Simulation Studies

**MvLSW simulation** ($P=6, Q=4, T=1024$):

| Setting | First half ($u < 0.5$) | Second half ($u > 0.5$) | Notes |
|---------|------------------------|--------------------------|-------|
| True coherence | Weak | Strong | Designed time-varying pattern |
| WaveCanCoh estimate | Accurately tracks weak dependence | Accurately tracks strong dependence | Verified over 1000 replications |
| 95% Wald CI | Covers true value | Covers true value | Reliable estimation |

**AR(2) mixture simulation** ($P=4, Q=3, T=1024$):

| Method | First half (shared gamma component) | Second half (no shared structure) | Notes |
|--------|--------------------------------------|-----------------------------------|-------|
| WaveCanCoh | Detects coherence | Accurately captures abrupt drop | Superior |
| LSP (Fourier-based) | Detects coherence | Fails to capture abrupt drop | Fails |

### LFP Data Analysis (Real Data)

Rat hippocampal LFP data, 22 electrodes split into two groups (T1/T2/T4/T5 vs T13–T17), at scale $j=5$ (15.625–31.25 Hz).

| Time point | Scale | Correct–error difference | $p$-value | Significance |
|------------|-------|--------------------------|-----------|--------------|
| $t^*=0.5$s | $j=4$ (31.25–62.5 Hz) | 0.023 | 0.001** | Significant |
| $t^*=0.5$s | $j=5$ (15.63–31.25 Hz) | 0.334 | 0.002** | Significant |
| $t^*=0.5$s | $j=6$ (7.81–15.63 Hz) | 0.039 | 0.001** | Significant |
| $t^*=1.0$s | $j=6$ | 0.002 | 0.025** | Significant |
| $t^*=-1.0$s | All scales | — | >0.1 | Not significant |

By contrast, the LSP (Fourier-based) method yields no significant differences under permutation testing on the same dataset.

### Key Findings

- **Correct vs. error trials**: Following odor stimulus onset ($t>0$), correct memory decision trials exhibit significantly higher inter-regional coherence in the 8–62 Hz band, whereas no significant difference is observed prior to stimulus onset.
- **Unequal channel contributions**: In correct trials, individual channels contribute more evenly to global coherence, whereas in error trials the coherence tends to be driven by a small number of dominant channels.
- WaveCanCoh demonstrates greater temporal sensitivity compared to Fourier-based methods.

## Highlights & Insights

- **Theoretical rigor**: A complete mathematical framework is presented, covering definitions, estimation, and consistency proofs, representing a solid theoretical contribution.
- **Interpretability**: Beyond providing a global coherence value, the framework reveals the time-varying contribution of each channel through canonical direction vectors.
- **Practical value**: Applied to neuroscience data, it uncovers behaviorally relevant neural coordination patterns that Fourier-based methods fail to detect.
- The causal extension (Causal-WaveCanCoh) offers a new tool for exploring directed interactions between brain regions.

## Limitations & Future Work

- Computational complexity grows with the number of channels due to LWS matrix inversion.
- The choice of smoothing parameter $M$ substantially affects estimation quality; adaptive selection methods are currently lacking.
- Only linear dependence is considered (an inherent limitation of canonical correlation); nonlinear dependence requires alternative frameworks.
- In high-dimensional settings (e.g., hundreds of electrodes), regularization or dimensionality reduction strategies may be necessary.

## Related Work & Insights

- Brillinger's (2001) frequency-domain canonical correlation is the direct antecedent that this work generalizes.
- The MvLSW model (Nason 2000, Ombao 2014) provides the theoretical foundation.
- The use of permutation testing ensures nonparametric statistical inference, which is methodologically elegant.
- The framework naturally extends to other nonstationary signal analysis domains such as finance (dynamic inter-sector correlations) and speech processing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First extension of canonical coherence to the wavelet domain for nonstationary settings
- Experimental Thoroughness: ⭐⭐⭐⭐ — Simulation validation + real neural data analysis + comparison with Fourier methods
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear; structure is well organized
- Value: ⭐⭐⭐⭐ — Broadly applicable to nonstationary multivariate signal analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[ICLR 2026\] Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](../../ICLR2026/time_series/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](channel_matters_estimating_channel_influence_for_multivariate_time_series.md)
- [\[NeurIPS 2025\] Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)

</div>

<!-- RELATED:END -->
