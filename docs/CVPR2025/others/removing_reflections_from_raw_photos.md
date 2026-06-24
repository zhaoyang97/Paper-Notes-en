---
title: >-
  [Paper Note] Removing Reflections from RAW Photos
description: >-
  [CVPR 2025][Reflection Removal] Proposes the first end-to-end reflection removal system based on RAW images. It simulates realistic reflections (including Fresnel, double reflection, white balance, and exposure) in the XYZ color space, trains an EfficientNet+BiFPN base model to separate the transmission and reflection layers, and then uses a Gaussian pyramid upsampler to preserve high-resolution details. An optional front-facing camera context map is leveraged to aid inferenc…
tags:
  - "CVPR 2025"
  - "Reflection Removal"
  - "RAW image"
  - "Photometric simulation"
  - "Upsampler"
  - "Dual camera"
date: 2026-05-08
content_hash: d49994cd3cf01fd5
---

# Removing Reflections from RAW Photos

**Conference**: CVPR 2025  
**arXiv**: [2404.14414](https://arxiv.org/abs/2404.14414)  
**Code**: Yes (Inference)  
**Area**: Others  
**Keywords**: Reflection Removal, RAW image, Photometric simulation, Upsampler, Dual camera

## TL;DR

Proposes the first end-to-end reflection removal system based on RAW images. It simulates realistic reflections (including Fresnel, double reflection, white balance, and exposure) in the XYZ color space, trains an EfficientNet+BiFPN base model to separate the transmission and reflection layers, and then uses a Gaussian pyramid upsampler to preserve high-resolution details. An optional front-facing camera context map is leveraged to aid inference, achieving a PSNR of 30.62 dB.

## Background & Motivation

### Background

Reflections that commonly occur when shooting through windows degrade image quality significantly. Existing reflection removal methods are trained on 8-bit JPG images; however, the physical process of window reflection (Fresnel reflectance, exposure, white balance, and tone mapping) is irreversibly compressed in JPG format.

### Limitations of Prior Work

(1) 8-bit JPGs discard precision in dark and highlight regions, which are critical for distinguishing reflections; (2) Synthetic training data lacks realism, as simple alpha blending fails to model Fresnel angle dependence, double-pane reflections, or color temperature disparities; (3) Upsampling after low-resolution predictions often reintroduces reflection artifacts.

### Key Challenge

Real-world reflection is a complex optical process, but training data is either unrealistic (synthetic) or lacks ground-truth (real).

### Key Insight

Simulate the complete physical process of reflection formation in the RAW domain (linear XYZ color space), and subsequently train the reflection removal model on RAW data, which preserves all photometric information.

### Core Idea

RAW domain physical simulation + context from a selfie camera + Gaussian pyramid upsampler = photometrically accurate reflection removal.

### Approach

**Goal**: ### Key Designs

1. **RAW Domain Physical Simulation**: Simulates in the XYZ space: Fresnel angle-dependent reflectance, multiple reflections from double-pane glass, different light source color temperatures (independent white balance for transmission and reflection), exposure differences, and blur.


## Method

### Overall Architecture


### Key Designs

1. **RAW Domain Physical Simulation**: Simulates the following in XYZ space: Fresnel angle-dependent reflectance, multiple reflections from double-pane glass, different light source color temperatures (independent white balance for transmission and reflection), exposure differences, and blur. Every simulation step is based on physical optics.

2. **Dual-Stream Base Model**: EfficientNet-B1 backbone + BiFPN fusion + StyleGAN-style mod-demod blocks. An optional second input (from a selfie camera capturing the indoor environment) provides context to help identify reflections.

3. **Gaussian Pyramid Upsampler**: Scales up predictions from 256p resolution to 2048p full resolution. A product mask based on feature matching is used to prevent reflections from being reintroduced during upsampling.

### Loss & Training

Base model: Perceptual loss (VGG19) + adversarial loss + gradient loss (5-tap) + L1 reflection loss. Upsampler: L1 (0.2) + L2 (0.2) + gradient (0.4) + LPIPS (0.8) + cycle-consistency (10.0). Trained entirely on synthetic data.

## Key Experimental Results

| Metric | Full System | Without RAW Simulation |
|------|---------|------------|
| PSNR (Transmission) | **30.62 dB** | Decreased by ~10 dB |
| SSIM | **95.2%** | Decreased by ~40pp |
| Context Map Gain | +4pp SSIM | — |

### Ablation Study

- RAW simulation vs. simplified simulation: Removing all physical components drops performance by 46pp, showing that RAW-domain simulation provides the most significant contribution.
- Context selfie image: +4pp SSIM (statistically significant, $p < 1.7 \times 10^{-11}$).
- Real-world deployment: 4.5 to 6.5 seconds of inference on MacBook / iPhone 14 Pro.

### Key Findings
- **RAW > JPG is the decisive factor**: Training in the RAW domain outperforms JPG by 40pp SSIM — a larger improvement than any architectural modification.
- **Every physical component of the simulation matters**: Removing gamma, exposure, Fresnel, or WB significantly degrades performance.
- **The upsampler's mask mechanism is effective**: It prevents reflections from "leaking" back during high-resolution restoration.

## Highlights & Insights
- **An exemplar of "Data > Model"**: Photometrically accurate simulation data yields a 40pp improvement, far exceeding architectural innovations.
- **End-to-end deployable**: Runs in real-time on an iPhone.

## Limitations & Future Work
- Requires RAW input (not applicable to compressed JPGs).
- Over-saturated regions require inpainting.
- Incapable of handling reflections with highly overlapping textures.

## Rating
- Novelty: ⭐⭐⭐⭐ RAW domain simulation and selfie assistance are key innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive component ablation and real-device deployment.
- Writing Quality: ⭐⭐⭐⭐ Detailed description of the physical simulation process.
- Value: ⭐⭐⭐⭐ Direct deployment potential for mobile camera applications.


## Related Work & Insights
- **vs Representative methods in the same field**: This work makes unique contributions to method design, complementing existing methods.
- **vs Traditional methods**: Compared to traditional solutions, the proposed method achieves significant improvements in key metrics.
- **Insights**: The technical approach of this work serves as an important reference for subsequent related research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](../../ICLR2026/others/predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)
- [\[CVPR 2026\] Zero-shot Detection of AI-Generated Image via RAW-RGB Alignment](../../CVPR2026/others/zero-shot_detection_of_ai-generated_image_via_raw-rgb_alignment.md)
- [\[CVPR 2025\] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)
- [\[CVPR 2025\] Integral Fast Fourier Color Constancy](integral_fast_fourier_color_constancy.md)
- [\[CVPR 2025\] PLeaS: Merging Models with Permutations and Least Squares](pleas_-_merging_models_with_permutations_and_least_squares.md)

</div>

<!-- RELATED:END -->
