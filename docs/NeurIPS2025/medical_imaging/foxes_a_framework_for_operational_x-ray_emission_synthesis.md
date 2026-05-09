---
title: >-
  [Paper Note] FOXES: A Framework For Operational X-ray Emission Synthesis
description: >-
  [NeurIPS 2025][Medical Imaging][Vision Transformer] This paper proposes FOXES, a Vision Transformer-based framework that translates multi-channel solar EUV observation images into soft X-ray (SXR) flux, achieving an overall Pearson correlation of 0.982. The framework lays the groundwork for far-side solar flare detection and the construction of more complete flare catalogs.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Vision Transformer
  - Solar Flares
  - EUV-SXR Translation
  - Space Weather
  - Attention Maps
  - Flare Classification
date: 2026-05-08
content_hash: 37911120326d9599
---

# FOXES: A Framework For Operational X-ray Emission Synthesis

**Conference**: NeurIPS 2025  
**arXiv**: [2510.22801](https://arxiv.org/abs/2510.22801)  
**Code**: Not provided  
**Area**: Solar Physics / Space Weather Forecasting  
**Keywords**: Vision Transformer, Solar Flares, EUV-SXR Translation, Space Weather, Attention Maps, Flare Classification

## TL;DR

This paper proposes FOXES, a Vision Transformer-based framework that translates multi-channel solar EUV observation images into soft X-ray (SXR) flux, achieving an overall Pearson correlation of 0.982. The framework lays the groundwork for far-side solar flare detection and the construction of more complete flare catalogs.

## Background & Motivation

**Solar Flares and Space Weather**: Solar flares are the radiative manifestation of rapid magnetic energy release in the corona, and serve as the best proxy indicators for coronal mass ejections (CMEs) and solar energetic particle events. When these events reach Earth, they can affect satellites, radio communications, GPS, and power grids.

**Limitations of Existing Flare Classification Methods**:
   - Classification relies primarily on integrated SXR irradiance from GOES satellites (A/B/C/M/X classes), but GOES measurements integrate over the full solar disk and cannot precisely localize the intensity and position of individual flares.
   - When multiple flares occur simultaneously (sympathetic flares), their individual contributions cannot be distinguished.

**Gap in Far-Side Monitoring**: No SXR monitors are currently deployed—or planned—beyond the Sun–Earth line, which severely limits the ability to protect far-side space activities such as Mars exploration missions from solar events.

**Advantages of EUV Imaging**: EUV images provide spatially resolved information, and missions such as STEREO and Solar Orbiter supply EUV observations from far-side vantage points. The ability to infer SXR flux from EUV data would therefore extend flare detection coverage.

## Method

### Data Preparation

**Input**: Six EUV channels from SDO/AIA (94, 131, 171, 193, 211, and 304 Å) at 1-minute cadence. Preprocessing is performed via the ITI pipeline: cropping to 1.1 solar radii, instrument degradation correction, image/exposure normalization, and downsampling to 512×512.

**Output**: 1-minute averaged SXR flux measurements from the GOES XRS 1–8 Å band.

**Data Splits** (no temporal overlap):

| Flare Class | Train | Validation | Test |
|-------------|-------|------------|------|
| < C | 13,256 | 358 | 13,370 |
| C | 48,140 | 15,722 | 63,503 |
| M | 16,604 | 2,827 | 22,256 |
| X | 1,738 | 81 | 1,075 |

Training period: January 2013–December 2022 + July 1–20, 2023; Test period: August 2023–September 2025.

### ViT Architecture

Built on a modified PyTorch Lightning ViT implementation:

- **Patch Size**: 16×16 → 1,024 patches
- **Embedding Dimension**: 512, jointly encoding multi-wavelength information across 6 EUV channels
- **Transformer Layers**: 6 layers, 8 attention heads
- **Feed-Forward Units**: 512
- **Dropout**: 0.1
- **Learning Rate**: 1e-4 with cosine annealing to 1e-5
- **Training**: 250 epochs, batch size 64
- **Loss Function**: Huber Loss, weighted by the inverse frequency of each flare class to mitigate class imbalance

### Physical Interpretation of Attention Maps

The self-attention maps of the ViT provide interpretable predictions—attention concentrates on physically relevant regions (active regions/flare sites) rather than spurious correlations. The left panel of Figure 3 shows attention heatmaps overlaid on the 131 Å channel, clearly pointing to the actual flare region.

## Key Experimental Results

### Quantitative Metrics (Log Space)

| Flare Class | MSE | RMSE | MAE | Pearson r |
|-------------|-----|------|-----|-----------|
| Overall | 1.41e-2 | 1.19e-1 | 8.97e-2 | **0.982** |
| < C | 9.52e-3 | 9.76e-2 | 9.22e-2 | 0.905 |
| C | 1.07e-2 | 1.03e-1 | 7.56e-2 | **0.962** |
| M | 2.56e-2 | 1.60e-1 | 1.26e-1 | 0.857 |
| X | 3.19e-2 | 1.79e-1 | 1.29e-1 | 0.665 |

### Key Findings

1. **Best performance on C-class flares**: The most abundant class, with MAE of only 7.56e-2 and a correlation of 0.962.
2. **Weakest performance on X-class flares**: $r = 0.665$, primarily due to data scarcity (only 1,738 training samples).
3. **Time-series tracking capability**: FOXES captures not only the impulsive and peak phases of flares but also accurately tracks the decay phase.
4. **Physical consistency of attention maps**: Highlighted attention regions align with observed active regions, particularly in the 131 Å channel, which is most sensitive to flare emission.

## Highlights & Insights

1. ⭐⭐⭐ **A new paradigm for EUV→SXR translation**: The first systematic application of ViT to translate multi-channel EUV images into SXR flux, with an overall Pearson $r = 0.982$ demonstrating proof of concept.
2. ⭐⭐ **Far-side application potential**: Directly applicable to far-side EUV observations from STEREO/Solar Orbiter, filling the flare detection gap beyond the Sun–Earth line.
3. ⭐⭐ **Physical interpretability**: Self-attention maps point to genuine active regions, suggesting the model has learned physically relevant features rather than statistical shortcuts.
4. ⭐ **Foundation for improved flare catalogs**: More comprehensive flare catalogs will in turn improve flare forecasting algorithms, creating a virtuous cycle.

## Limitations & Future Work

1. **Predicts only integrated flux**: The current model outputs only full-disk integrated SXR flux and cannot localize the position or intensity of individual flares—an explicitly stated future direction (patch-based SXR prediction).
2. **Insufficient X-class performance**: A correlation of $r = 0.665$ for the highest-risk events (the class most in need of accurate prediction) falls far short, primarily due to class imbalance.
3. **No comparison with alternative methods**: The paper does not conduct systematic comparisons with existing EUV–SXR translation approaches or CNN baselines.
4. **Cross-instrument calibration unvalidated**: The framework claims applicability to STEREO but requires ITI calibration; cross-instrument generalization has not been empirically verified.
5. **Absence of temporal modeling**: The model processes each time-step snapshot independently, without exploiting the continuity of the time series.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Technical Depth | ⭐⭐ |
| Experimental Thoroughness | ⭐⭐ |
| Writing Quality | ⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐ |

## Related Work & Insights

## Rating

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Modeling X-ray Photon Pile-up with a Normalizing Flow](modeling_x-ray_photon_pile-up_with_a_normalizing_flow.md)
- [\[NeurIPS 2025\] Variational Autoencoder with Normalizing Flow for X-ray Spectral Fitting](variational_autoencoder_with_normalizing_flow_for_x-ray_spectral_fitting.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)

<!-- RELATED:END -->
