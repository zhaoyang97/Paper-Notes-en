---
title: >-
  [Paper Note] Transformer Embeddings for Fast Microlensing Inference
description: >-
  [NeurIPS 2025][Time Series][Microlensing] This paper combines a Transformer encoder with Neural Posterior Estimation (NPE) to perform fast, well-calibrated parameter inference directly from sparse, noisy, and irregularly sampled microlensing light curves, achieving speedups exceeding $10^4\times$ over traditional MCMC methods.
tags:
  - NeurIPS 2025
  - Time Series
  - Microlensing
  - Simulation-Based Inference
  - Transformer
  - Posterior Estimation
  - Free-Floating Planets
date: 2026-05-08
content_hash: afdaf4c1df0f0ceb
---

# Transformer Embeddings for Fast Microlensing Inference

**Conference**: NeurIPS 2025
**arXiv**: [2512.11687](https://arxiv.org/abs/2512.11687)
**Code**: [GitHub](https://github.com/NolanSmyth/sbi_microlensing_transformers)
**Area**: Astronomy, Time Series Inference
**Keywords**: Microlensing, Simulation-Based Inference, Transformer, Posterior Estimation, Free-Floating Planets

## TL;DR

This paper combines a Transformer encoder with Neural Posterior Estimation (NPE) to perform fast, well-calibrated parameter inference directly from sparse, noisy, and irregularly sampled microlensing light curves, achieving speedups exceeding $10^4\times$ over traditional MCMC methods.

## Background & Motivation

- **State of the Field**: Free-floating planets (FFPs) may be among the most abundant Earth-mass exoplanets, and microlensing is the most promising technique for detecting them.
- **Limitations of Prior Work**: The Nancy Grace Roman Space Telescope is expected to detect thousands of FFPs, demanding rapid signal characterization at scale. Traditional MCMC methods are computationally prohibitive and cannot scale to Roman's billions of light curves. Prior RNN-based approaches suffer **catastrophic failure** upon the introduction of minor data gaps, exposing the classical challenge of distributional shift in time series.
- **Root Cause**: Conventional sequential models are ill-suited for irregularly sampled, variable-length, and sparse time series data.
- **Paper Goals**: Simulation-based inference (SBI) provides an amortized posterior estimation framework — train once, infer extremely fast. The self-attention mechanism of Transformers naturally accommodates irregular sampling, variable lengths, and sparse observations.

## Method

### Overall Architecture

$$\theta \rightarrow \text{Simulated Light Curves} \rightarrow \text{Data Augmentation (gaps, dropout, noise)} \rightarrow \text{Transformer Encoder} \rightarrow \text{Embedding } z \rightarrow \text{Normalizing Flow} \rightarrow p(\theta|x)$$

### Key Designs

**Physical Model**: Finite Source Point Lens (FSPL) with 5 parameters:
- $t_0$: time of closest approach
- $u_0$: minimum impact parameter
- $t_E$: Einstein crossing time
- $\rho$: normalized source radius
- $f_s$: source flux fraction

**Data Augmentation** (online, on-the-fly):
- Seasonal gaps: 0–3 gaps, each lasting 1–10 days
- Random dropout: 0%–60% of data points removed
- Noise injection: Gaussian photometric noise with $\sigma \in [0.001, 0.02]$

**Network Architecture**:
- Input: sequences padded to $L = 1000$, with 3 channels per timestep $(t_\text{norm},\, F,\, \sigma)$
- Transformer encoder: 6 layers, 8 heads, 256-dimensional embeddings, 512-dimensional FFN
- Sinusoidal positional encoding + masked mean pooling for aggregation
- Posterior estimator: Masked Autoregressive Flow (MAF)

**Recoverability Filtering**: at least 5 points within $t_E/2$ of peak, at least 5 points more than $2t_E$ from peak, and peak magnification exceeding $5\times$ the mean noise level.

### Loss & Training

- 80,000 simulated events for training + 20,000 for validation
- Adam optimizer, initial learning rate $10^{-4}$, ReduceLROnPlateau (factor 0.5, patience 10 epochs)
- Training completed in approximately 20 hours on a single Nvidia H100 GPU

## Key Experimental Results

### Main Results: Calibration on Simulated Data

| Parameter | Recoverability |
|-----------|---------------|
| $t_0$ | Well recovered across full range |
| $u_0$ | Well recovered when $u_0 > \rho$ (point-source regime) |
| $t_E$ | Well recovered |
| $\rho$ | Well recovered when $u_0 < \rho$ (finite-source effects prominent) |
| $f_s$ | Well recovered |

### Speed Comparison

| Method | Time to Generate 15,000 Samples | Speedup |
|--------|----------------------------------|---------|
| NPE (GPU) | 0.08 s | $> 10^4\times$ |
| NPE (CPU) | 0.82 s | $\sim 1.2 \times 10^3\times$ |
| MCMC (CPU) | 959 s | Baseline |

### Real-Data Validation (KMT-2019-BLG-2073)

| Parameter | NPE Recovery | Literature Value |
|-----------|-------------|-----------------|
| $t_0$ | $8708.60 \pm 0.02$ | $8708.58$ |
| $u_0$ | $0.20 \pm 0.11$ | $0.32$ |
| $t_E$ | $0.355 \pm 0.03$ | $0.50$ |
| $\rho$ | $0.832 \pm 0.09$ | N/A |
| $f_s$ | $0.82 \pm 0.13$ | $0.61$ |

### Key Findings

- TARP diagnostics confirm that posterior estimates are **well calibrated**.
- NPE posteriors agree with MCMC results with only mild broadening, an inherent property of amortized inference.
- The FSPL model provides better fits near peak than PSPL, with smaller residuals.
- Parameter discrepancies relative to the literature likely stem from differences in photometric extraction pipelines (pySIS vs. TLC).

## Highlights & Insights

- **Strong Practicality**: The method operates directly on raw time series without interpolation or complex preprocessing.
- Once trained, inference is extremely fast, making the approach well suited for large-scale data processing with the Roman Space Telescope.
- Online data augmentation renders the model robust to diverse data quality issues.
- Masked mean pooling provides a simple yet effective mechanism for handling variable-length sequences.

## Limitations & Future Work

- Training relies solely on Gaussian noise; systematic noise sources and false-positive signals (e.g., stellar variability) are not modeled.
- A fixed 20-day window limits applicability to long-timescale events.
- SBI is sensitive to prior specification, requiring careful calibration against Galactic and lens population models.
- The contribution of individual Transformer components and the optimal model size are not ablated.
- Validation is performed on only one real event; broader evaluation is needed.

## Related Work & Insights

- Compared to the microlensing SBI work of Zhang et al. (using 1D ResNet + GRU), the key improvement lies in the natural handling of irregular sampling.
- This work aligns with the broader trend of applying Transformers to astronomical time series (e.g., MAVEN, SpectraFM).
- The proposed method complements anomaly detection pipelines — SBI for characterization, anomaly detection for candidate screening.

## Rating

⭐⭐⭐⭐ — A practical and effective method with a clear astronomical application scenario, significant speedup, and excellent calibration quality.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Decomposition of Small Transformer Models](decomposition_of_small_transformer_models.md)
- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)
- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](../../ICLR2026/time_series/relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)
- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [\[NeurIPS 2025\] BubbleFormer: Forecasting Boiling with Transformers](bubbleformer_forecasting_boiling_with_transformers.md)

<!-- RELATED:END -->
