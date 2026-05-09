---
title: >-
  [Paper Note] A Real-world Display Inverse Rendering Dataset
description: >-
  [ICCV 2025][LLM Evaluation][Inverse Rendering] This paper presents the first real-world inverse rendering dataset (DIR) built upon an LCD display–polarization camera system, comprising polarimetric stereo images of objects with diverse reflectance properties captured under OLAT illumination, calibrated display backlight/nonlinearity, and high-quality ground-truth geometry. A simple yet effective baseline method for display-based inverse rendering is also proposed.
tags:
  - ICCV 2025
  - LLM Evaluation
  - Inverse Rendering
  - Display Light Source
  - Polarization Imaging
  - Dataset
  - Diffuse-Specular Separation
date: 2026-05-08
content_hash: 2791ddc3ab8a28f4
---

# A Real-world Display Inverse Rendering Dataset

**Conference**: ICCV 2025
**arXiv**: [2508.14411](https://arxiv.org/abs/2508.14411)
**Code**: [https://michaelcsj.github.io/DIR/](https://michaelcsj.github.io/DIR/)
**Area**: LLM Evaluation
**Keywords**: Inverse Rendering, Display Light Source, Polarization Imaging, Dataset, Diffuse-Specular Separation

## TL;DR
This paper presents the first real-world inverse rendering dataset (DIR) built upon an LCD display–polarization camera system, comprising polarimetric stereo images of objects with diverse reflectance properties captured under OLAT illumination, calibrated display backlight/nonlinearity, and high-quality ground-truth geometry. A simple yet effective baseline method for display-based inverse rendering is also proposed.

## Background & Motivation

**Background**: Inverse rendering aims to recover geometry and reflectance from images. Existing imaging systems include light stages (LED spherical arrays), flash photography, and display–camera systems. The former two are either prohibitively expensive or require multiple captures with a moving camera.

**Limitations of Prior Work**: Display–camera systems offer unique advantages — each pixel can serve as a programmable point light source, and the polarized light emitted by LCDs facilitates diffuse-specular separation. Despite this potential, no public display–camera inverse rendering dataset exists, severely impeding research progress in this direction. All existing datasets are collected using other systems such as light stages, light probes, or robotic arms.

**Key Challenge**: Display-based inverse rendering presents unique challenges including near-field illumination, low light power (as little as 0.06 mcd per pixel), backlight leakage, polarization effects, and non-uniform angular sampling, necessitating dedicated datasets and methods.

**Goal**: (1) Construct and calibrate a display–polarization camera imaging system; (2) collect the first high-quality real-world display inverse rendering dataset; (3) evaluate existing methods and provide baselines.

**Key Insight**: Exploit two key properties of LCDs — programmability (OLAT illumination) and polarization (diffuse-specular separation) — combined with structured-light scanning for ground-truth geometry acquisition.

**Core Idea**: Systematically address the data scarcity problem in display-based inverse rendering by calibrating display backlight and nonlinearity, designing 144-superpixel OLAT patterns, employing a polarization camera to separate diffuse/specular components, and obtaining ground-truth geometry via structured-light scanning.

## Method

### Overall Architecture
The imaging system consists of a Samsung Odyssey Ark LCD display paired with a dual polarimetric RGB camera (FLIR BFS), capturing polarimetric stereo image pairs of diverse objects under 144 OLAT superpixel illumination patterns. Each object is accompanied by ground-truth geometry obtained via structured-light scanning.

### Key Designs

1. **Display Backlight and Nonlinearity Calibration**:

    - Function: Accurately model the actual light output of the display.
    - Mechanism: An LCD exhibits non-negligible backlight leakage even when set to black, and the luminance–setpoint relationship is nonlinear. This is modeled as $L_i = s(P_i + B_i)^\gamma$, where $B_i$ is the spatially varying backlight and $\gamma$ is the nonlinearity exponent. Calibration parameters are jointly optimized using a sphere of known geometry and reflectance captured under OLAT illumination.
    - Design Motivation: Failing to calibrate the backlight introduces systematic errors across all OLAT images, significantly degrading inverse rendering accuracy.

2. **Polarimetric Diffuse-Specular Separation**:

    - Function: Decompose each captured image into diffuse and specular components.
    - Mechanism: LCDs emit vertically polarized light, and the polarization camera simultaneously captures images at four polarization angles (0°/45°/90°/135°). Leveraging polarization properties — specular reflections preserve polarization while diffuse reflections depolarize — the two components are separated.
    - Design Motivation: Diffuse and specular components obey different BRDF models; separating them enables more accurate estimation of normals and reflectance.

3. **OLAT + Linear Combination for Flexible Relighting**:

    - Function: Support image synthesis under arbitrary display illumination patterns.
    - Mechanism: By the linear superposition principle of light, images under any display pattern can be synthesized as a linear combination of OLAT basis images. The dataset also supports simulation at different noise levels.
    - Design Motivation: The 144 OLAT images serve as "illumination basis functions," greatly extending the flexibility of the dataset.

### Loss & Training
The baseline method employs a physics-aware optimization strategy: leveraging the calibrated near-field illumination model and polarimetric separation, it iteratively optimizes normal and reflectance parameters.

## Key Experimental Results

### Main Results

| Method | Normal Estimation | Reflectance Estimation | Near-field Handling |
|---|---|---|---|
| Traditional Photometric Stereo | Moderate | N/A | Poor (far-field assumption) |
| NeRF-based Inverse Rendering | Moderate | Fair | Poor |
| Ours (Baseline) | **Best** | **Best** | **Good (explicit near-field modeling)** |

### Ablation Study

| Configuration | Performance | Note |
|---|---|---|
| w/o Backlight Calibration | Significant drop | Backlight leakage introduces systematic error |
| w/o Polarimetric Separation | Drop | Inaccurate BRDF modeling |
| Far-field Illumination Assumption | Drop | Near-field effects are non-negligible |
| Full Calibration + Polarization | Best | Accurate modeling is critical |

### Key Findings
- Existing inverse rendering methods all perform poorly under the display setting — near-field illumination effects are the primary cause.
- Backlight calibration is critical for accurate reconstruction — uncalibrated backlight leads to several-fold increase in normal estimation error.
- Polarimetric component separation substantially improves reflectance estimation for specular objects.
- A simple baseline that accounts for near-field effects outperforms all SOTA methods.

## Highlights & Insights
- **Filling a Data Gap**: The first real-world display inverse rendering dataset, establishing a research foundation for a promising yet benchmark-deficient field.
- **Importance of Backlight Calibration**: Reveals a previously overlooked practical issue — LCD backlight leakage is a non-negligible error source in inverse rendering.
- **Low Cost, High Quality**: Compared to light stages (costing hundreds of thousands of dollars), the display system is extremely affordable, potentially democratizing inverse rendering research.

## Limitations & Future Work
- Display luminance is limited (0.06 mcd per superpixel), making low SNR an inherent challenge.
- The 144-superpixel angular sampling is rather coarse, limiting normal map resolution.
- The current range of object types and quantities is limited and needs expansion.
- Time-multiplexing strategies could be explored to improve SNR.

## Related Work & Insights
- **vs. Light Stage Datasets (e.g., OpenIllumination)**: Higher precision but extremely costly and non-portable; the proposed dataset is low-cost and supports polarimetric separation.
- **vs. Choi et al. (3D-printed objects)**: Insufficient material diversity; the proposed dataset uses real objects covering diverse materials.
- The systematic calibration methodology for display-based inverse rendering is generalizable to other near-field illumination systems.

## Rating
- Novelty: ⭐⭐⭐⭐ First dataset of its kind, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes comparative evaluation of multiple existing methods.
- Writing Quality: ⭐⭐⭐⭐ System design and calibration procedures are described in thorough detail.
- Value: ⭐⭐⭐⭐ The dataset and calibration methodology offer lasting value to the display inverse rendering community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing](phatnet_a_physics-guided_haze_transfer_network_for_domain-adaptive_real-world_im.md)
- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](forcennet_foreground-centric_network_for_document_image_rectification.md)
- [\[ICCV 2025\] Supercharging Floorplan Localization with Semantic Rays](supercharging_floorplan_localization_with_semantic_rays.md)
- [\[ICCV 2025\] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](odp-bench_benchmarking_out-of-distribution_performance_prediction.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)

<!-- RELATED:END -->
