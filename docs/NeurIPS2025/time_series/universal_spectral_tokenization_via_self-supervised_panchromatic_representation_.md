---
title: >-
  [Paper Note] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning
description: >-
  [NeurIPS 2025][Time Series][Spectral Tokenizer] This paper proposes the first universal spectral tokenizer that jointly trains on heterogeneous astronomical spectra (SDSS/DESI/GALAH/APOGEE) on their native wavelength grids via continuous wavelength embeddings and self-supervised reconstruction objectives, producing aligned, uniform, and physically meaningful representations.
tags:
  - NeurIPS 2025
  - Time Series
  - Spectral Tokenizer
  - Heterogeneous Data Unification
  - Vision Transformer
  - Self-Supervised Pretraining
  - Astronomy
date: 2026-05-08
content_hash: 33e784f2fe45f611
---

# Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.17959](https://arxiv.org/abs/2510.17959)
**Code**: N/A
**Area**: Astronomical Spectroscopy, Foundation Models, Self-Supervised Learning
**Keywords**: Spectral Tokenizer, Heterogeneous Data Unification, Vision Transformer, Self-Supervised Pretraining, Astronomy

## TL;DR
This paper proposes the first universal spectral tokenizer that jointly trains on heterogeneous astronomical spectra (SDSS/DESI/GALAH/APOGEE) on their native wavelength grids via continuous wavelength embeddings and self-supervised reconstruction objectives, producing aligned, uniform, and physically meaningful representations.

## Background & Motivation

### Root Cause

**Root Cause**: Large astronomical surveys (SDSS, DESI, etc.) have collected millions of spectra, yet these cover different wavelength ranges and spectral resolutions.

### State of the Field

**State of the Field**: Existing analysis pipelines are fragmented: each survey requires independent preprocessing and task-specific models, precluding cross-survey knowledge sharing.

### Limitations of Prior Work

**Limitations of Prior Work**: Fixed-grid methods for unifying multi-resolution data introduce interpolation artifacts, and are computationally infeasible over broad wavelength ranges (requiring ~300K pixels).

### Starting Point

**Starting Point**: The central challenge for scientific foundation models is learning universal representations from irregular, multi-resolution sequential data.

## Method

### Overall Architecture
- Based on the Vision Transformer (ViT) architecture, adapted for one-dimensional spectral data processing.
- Encoder: receives spectral data on native wavelength grids and produces uniform wavelength-aware embeddings.
- Decoder: reconstructs the original spectrum from embeddings, supporting arbitrary output wavelength grids.
- Self-supervised pretraining with lightweight downstream adaptation.

### Key Designs
1. **Continuous Wavelength Embedding**:

    - Applies per-pixel sinusoidal positional encoding $PE(\lambda)_k$, with frequencies $\omega_k$ evenly spaced in log space.
    - Operates directly on native wavelength grids without resampling or interpolation.
    - Wavelength embeddings are added to flux patches to inject positional wavelength information.

2. **Heterogeneous Input Handling**:

    - Spectral normalization: divide by median flux to focus on relative variations.
    - Patch-level masking: patches with more than half bad pixels are flagged as invalid.
    - Bad patches are automatically excluded during attention computation.

3. **Loss-Aware Reconstruction**:

    - The decoder receives sinusoidal embeddings of the target wavelength grid as additional input.
    - Gaussian likelihood reconstruction loss computed only over valid pixels.
    - $\mathcal{L} = \frac{1}{N}\sum_i m_i \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}$
    - Weighted by measurement uncertainty $\sigma_i$, giving greater contribution to high-SNR pixels.

### Loss & Training
- Encoder: 6 layers; Decoder: 6 layers; embedding dimension 512; 8 attention heads.
- Patch size: 32 pixels; batch size: 64.
- AdamW optimizer, learning rate 1e-4, trained for 600k steps.
- 4× NVIDIA A100-SXM4-40GB GPUs, 48 hours of training.

## Key Experimental Results

### Training Data Overview

| Dataset | Wavelength Range | Resolution | Object Type |
|--------|-----------------|-----------|------------|
| SDSS DR17 | 3600–10400 Å | R~2000 | Galaxies/Quasars/Stars |
| DESI DR1 | 3600–9800 Å | R~5000 | Galaxies/Quasars/Stars |
| GALAH DR3 | 4700–7900 Å | R~28000 | Stars |
| APOGEE | 1.51–1.7 μm | R~22500 | Stars |

### Main Results

**Object Classification (DESI Spectra)**

| Model | Galaxy | Quasar | Star | Average |
|-------|--------|--------|------|---------|
| Zhong et al. (specialized model) | 93% | 99% | 98% | 96% |
| Ours + Adaptation Module | 94% | 97% | 98% | **96%** |

**Stellar Parameter Estimation (APOGEE Spectra)**

| Model | log g | T_eff | [Fe/H] |
|-------|-------|-------|--------|
| The Cannon 2 | 0.07 dex | 38 K | 0.03 dex |
| astroNN | 0.05 dex | 30 K | 0.02 dex |
| Olney et al. | 0.15 dex | 100 K | 0.07 dex |
| Ours + Adaptation Module | 0.07 dex | **23 K** | 0.02 dex |

### Key Findings
- A single model achieves unified representation across 4 surveys, spanning optical/infrared wavelengths and stars/galaxies/quasars.
- The unsupervisedly learned embedding space naturally exhibits physical structure: UMAP visualizations reveal clear gradients of stellar mass and redshift.
- A lightweight adaptation module (frozen encoder + linear layer) is sufficient to achieve competitive downstream performance against specialized baselines.
- Reconstruction quality spans multiple orders of magnitude in flux and covers diverse physical phenomena.

## Highlights & Insights
- This work represents the first unified spectral model across surveys, resolutions, and object types, carrying significant methodological importance.
- The continuous wavelength embedding elegantly circumvents the limitations of fixed grids and naturally generalizes to other irregular sequential data such as time series.
- The pretrained representations operate without redshift information, breaking the circular dependency of "estimate redshift before analysis."
- The domain-agnostic architectural design positions this work as a potential building block for scientific foundation models.

## Limitations & Future Work
- More advanced pretraining objectives such as masked autoencoding or contrastive learning have not been explored.
- Downstream tasks use simple mean pooling, discarding intra-sequence wavelength-dependent information.
- Cross-survey transfer learning (e.g., training on one survey and evaluating on another) has not been demonstrated.
- The model scale is relatively small (6+6 layers); performance gains from larger models remain to be explored.

## Related Work & Insights
- The use of continuous positional encodings for irregular grids is generalizable to medical signals, climate data, and beyond.
- The paradigm of a single encoder with multiple downstream adapters holds broad promise for scientific data applications.
- Measurement-error-weighted reconstruction loss offers a valuable reference for handling noisy scientific data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Both the problem formulation and proposed solution are pioneering)
- Technical Contribution: ⭐⭐⭐⭐ (Universal architecture design is concise and effective)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated across multiple datasets and tasks)
- Writing Quality: ⭐⭐⭐⭐ (Motivation is clear and presentation is thorough)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] WaLRUS: Wavelets for Long-range Representation Using SSMs](walrus_wavelets_for_long-range_representation_using_ssms.md)
- [\[NeurIPS 2025\] Fern: Chaining Spectral Pearls — Ellipsoidal Forecasting Beyond Trajectories for Time Series](friren_beyond_trajectories_--_a_spectral_lens_on_time.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[ICLR 2026\] Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model](../../ICLR2026/time_series/brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled.md)
- [\[AAAI 2026\] iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning](../../AAAI2026/time_series/beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)

<!-- RELATED:END -->
