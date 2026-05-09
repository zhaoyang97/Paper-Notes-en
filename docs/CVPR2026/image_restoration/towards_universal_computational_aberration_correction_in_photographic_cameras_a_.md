---
title: >-
  [Paper Note] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis
description: >-
  [CVPR 2026][Image Restoration][aberration-correction] This paper constructs UniCAC, the first universal computational aberration correction benchmark for consumer-grade cameras, proposes an Optical Degradation Evaluator (ODE) to quantify aberration difficulty, systematically evaluates 24 image restoration/CAC methods, and reveals three key factors influencing CAC performance.
tags:
  - CVPR 2026
  - Image Restoration
  - aberration-correction
  - benchmark
  - optical-degradation
  - computational-imaging
  - lens-design
date: 2026-05-08
content_hash: 3576a6f2a7b2afb0
---

# Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis

**Conference**: CVPR 2026
**arXiv**: [2603.12083](https://arxiv.org/abs/2603.12083)
**Code**: None
**Area**: Image Restoration
**Keywords**: aberration-correction, benchmark, optical-degradation, computational-imaging, lens-design

## TL;DR

This paper constructs UniCAC, the first universal computational aberration correction benchmark for consumer-grade cameras, proposes an Optical Degradation Evaluator (ODE) to quantify aberration difficulty, systematically evaluates 24 image restoration/CAC methods, and reveals three key factors influencing CAC performance.

## Background & Motivation

Computational aberration correction (CAC) is a post-processing technique for optical imaging systems that corrects residual optical aberrations. Existing CAC methods suffer from the following limitations:

1. **System-specific design**: Existing methods are designed for specific lenses, with poor generalization, requiring time-consuming retraining for new lenses.
2. **Lack of comprehensive benchmarks**: Existing benchmarks lack datasets covering sufficiently diverse optical aberrations, making it difficult to evaluate cross-lens generalizability.
3. **Insufficient aberration quantification metrics**: Traditional metrics such as RMS radius exhibit weak linear correlation with downstream CAC performance.
4. **Unclear influencing factors**: Which factors—prior utilization, network architecture, training strategy—most significantly affect CAC performance remains unclear.

## Method

### 1. Automated Optical Design for Lens Library Generation

The OptiFusion method is extended by redefining spherical parameter definitions to include aspherical parameters, enabling automated design of a large number of spherical and aspherical lenses. Four categories of specification variables are considered: number of lens elements, aperture stop position, half field of view, and F-number, ensuring diversity in aberration characteristics.

### 2. Optical Degradation Evaluator (ODE)

The ODE comprehensively assesses optical degradation along three dimensions:

$$ODE = \lambda_{oiq} \cdot OIQ + \lambda_s \cdot U_s + \lambda_c \cdot U_c$$

where:

**Optical Image Quality (OIQ)**: integrates traditional IQA metrics and MTF-based optical evaluation:

$$OIQ = \alpha \frac{PSNR}{50} + \beta \frac{SSIM - 0.5}{0.5} + \gamma \cdot OIQE$$

with $\alpha=0.4, \beta=0.3, \gamma=0.3$.

**Spatial Uniformity ($U_s$) and Chromatic Uniformity ($U_c$)**: measured using the coefficient of variation of OIQ values:

$$U_{s,c} = e^{-\sigma \cdot CV_{s,c}}$$

$U_s$ is computed from 5 field positions, and $U_c$ from 3 color channels.

Final weights: $\lambda_{oiq}=0.7, \lambda_s=0.3, \lambda_c=0.01$.

### 3. Overall Performance Metric (O.P.)

$$O.P. = 4 \times \frac{PSNR}{50} + 3 \times \frac{SSIM-0.5}{0.5} + 4 \times \frac{1-LPIPS}{0.4} + 3 \times OIQE + 1 \times \frac{100-FID}{100} + 1 \times ClipIQA$$

This covers three dimensions: image fidelity, optical quality, and perceptual quality.

### 4. Benchmark Construction

- Lens library: 873 training lenses + 120 test lenses
- Training set: ~3,000 GT images (Flickr2K + DIV2K), degraded using PSFs randomly sampled from training lenses
- Test set: 26 self-captured high-resolution GT images, degraded using PSFs from 120 test lenses
- Lenses are categorized into 5 degradation levels based on ODE

## Key Experimental Results

### Overall Performance Ranking — Top-5 among 24 Methods

| Rank | Method | Type | PSNR↑ | SSIM↑ | LPIPS↓ | OIQE↑ | O.P.↑ |
|------|--------|------|-------|-------|--------|-------|-------|
| 1 | FeMaSR | IR-GAN | 26.94 | 0.841 | 0.136 | 0.722 | 1.618 |
| 2 | NAFNet | IR-Reg | 27.78 | 0.876 | 0.211 | 0.705 | 1.549 |
| 3 | DiffBIR | IR-Diff | 27.65 | 0.812 | 0.196 | 0.711 | 1.547 |
| 4 | MIMOUNet | IR-Reg | 27.36 | 0.870 | 0.229 | 0.742 | 1.527 |
| 6 | FOV-KPN | CAC | 26.34 | 0.824 | 0.184 | 0.631 | 1.502 |

### Key Observations Across Degradation Levels

| Aberration Level | FeMaSR Rank | DiffBIR Rank | FOV-KPN Rank | PART Rank |
|-----------------|-------------|--------------|--------------|-----------|
| L1 (Mild) | High | Medium | High | Medium |
| L5 (Severe) | High | Rising | Declining | Rising |

### Nine Key Findings

**Prior Utilization**: (1) Optical priors (field-of-view information, PSF cues) are critical for handling spatially varying aberrations; (2) clean image priors (codebook/diffusion priors) are highly beneficial for CAC.

**Network Architecture**: (3) CNNs offer a better CAC performance–speed trade-off, as convolution effectively captures local features that match the nature of aberration degradation.

**Training Strategy**: (4) Regression-based training enhances image fidelity; (5) GAN/diffusion-based training improves perceptual quality; (6) training strategies targeting optical quality (OIQE) remain to be explored.

## Highlights & Insights

- **First comprehensive universal CAC benchmark**, covering both spherical and aspherical lenses across a full aberration distribution.
- The **ODE framework** exhibits higher linear correlation with CAC performance ($R^2$) compared to traditional RMS radius metrics.
- **Systematic evaluation of 24 methods** yields 9 actionable key findings for the community.
- Aberration simulation demonstrates high consistency with Zemax simulations and real captured images.
- Complete Zemax files and code are provided to the community.

## Limitations & Future Work

- Test lens PSF degradation is generated through simulation, and gaps with real cameras may exist (ISP pipeline, noise models, etc. are not considered).
- Only refractive consumer-grade camera lenses are addressed; diffractive optical elements and reflective systems are not covered.
- All evaluated methods adopt a unified training configuration; training strategies specifically optimized for CAC are not explored.
- $U_c$ shows weak correlation with CAC performance but is retained in ODE, potentially introducing noise.

## Rating

⭐⭐⭐⭐ — As a benchmark paper, the contributions are solid and comprehensive: the dataset construction is scientifically rigorous, the evaluation framework is novel and well-validated, and the systematic evaluation of 24 methods provides valuable guidance for the community. However, methodological novelty is limited, with the work being more oriented toward empirical analysis.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] UniCAC: Towards Universal Computational Aberration Correction in Photographic Cameras](unicac_universal_computational_aberration_correction.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)

<!-- RELATED:END -->
