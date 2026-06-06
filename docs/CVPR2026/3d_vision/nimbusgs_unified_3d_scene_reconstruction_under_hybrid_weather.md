---
title: >-
  [Paper Note] NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] NimbusGS proposes a unified 3D scene reconstruction framework that decomposes weather degradation into a continuous scattering field (fog/haze) and a per-view particulate res…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "adverse weather"
  - "scene reconstruction"
  - "physical modeling"
  - "weather decomposition"
date: 2026-05-08
content_hash: 1b84201890fcd00a
---

# NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather

**Conference**: CVPR 2026  
**arXiv**: [2603.27228](https://arxiv.org/abs/2603.27228)  
**Code**: [https://github.com/lyy-ovo/NimbusGS](https://github.com/lyy-ovo/NimbusGS)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, adverse weather, scene reconstruction, physical modeling, weather decomposition

## TL;DR

NimbusGS proposes a unified 3D scene reconstruction framework that decomposes weather degradation into a continuous scattering field (fog/haze) and a per-view particulate residual layer (rain/snow), coupled with a geometry-guided gradient scaling mechanism, achieving state-of-the-art reconstruction under individual and hybrid weather conditions within a single framework.

## Background & Motivation

3D scene reconstruction conventionally assumes clean, high-quality inputs; however, real-world weather conditions such as fog, rain, and snow severely compromise image formation. Weather degradation manifests through two distinct physical mechanisms: (1) continuous media (fog/haze)—depth-dependent light attenuation that is consistent across views; and (2) discrete particles (rain/snow)—dynamic, high-frequency occlusions that are independent across views.

**Limitations of Prior Work**: Two-stage pipelines that apply image restoration prior to reconstruction destroy multi-view consistency; methods that embed weather modeling into reconstruction typically address only a single weather type. Existing approaches universally fail under hybrid weather conditions (e.g., simultaneous fog and rain).

**Core Problem**: Grounded in the physical nature of weather, this work designs a unified framework that jointly models both continuous scattering and discrete particle degradation mechanisms.

## Method

### Overall Architecture

Built upon 3DGS, the framework introduces two degradation modeling branches: a continuous scattering field that estimates global transmittance and atmospheric light (view-consistent), and a particulate residual layer that captures per-view dynamic interference (view-independent). A geometry-guided gradient scaling mechanism stabilizes the optimization process.

### Key Designs

1. **Continuous Scattering Field**:

    - Function: Models view-consistent light attenuation caused by continuous media such as fog and haze.
    - Mechanism: A volumetric extinction field estimates scene-level transmittance and atmospheric light. Transmittance decays with depth, and atmospheric light is shared across all views. The rendering equation extends standard 3DGS with an atmospheric scattering model.
    - Design Motivation: Fog and haze are global physical phenomena that must be modeled in a view-consistent manner to preserve multi-view geometric consistency.

2. **Particulate Residual Layer**:

    - Function: Models view-independent local occlusions caused by discrete particles such as rain and snow.
    - Mechanism: An independent residual map is maintained per view to capture transient interference specific to that viewpoint. Residuals are composited after rendering without affecting the underlying 3D geometry. During training, the model automatically learns to assign transient interference to the residual layer and persistent structures to the Gaussian field.
    - Design Motivation: Raindrops and snowflakes occupy different positions across views and cannot be modeled in a view-consistent manner, necessitating per-view independent treatment.

3. **Geometry-Guided Gradient Scaling**:

    - Function: Stabilizes geometry learning under severe occlusion.
    - Mechanism: Gradient magnitudes are adaptively scaled according to visibility cues across different regions. Regions with high visibility receive normal gradients, while regions with low visibility (heavily occluded by fog or haze) have their gradients attenuated to prevent noise from dominating optimization. This resolves gradient imbalance in distant or severely degraded regions.
    - Design Motivation: Weather induces non-uniform visibility; distant, heavily degraded regions carry weak reconstruction signals yet may exhibit large gradients due to noise, requiring explicit suppression.

### Loss & Training

A progressive optimization strategy is employed to gradually decouple continuous scattering from particulate effects. The training objective comprises an L1 reconstruction loss, D-SSIM, and regularization losses for the scattering field and residual layer. No paired data or large-scale pretraining is required.

## Key Experimental Results

### Main Results

| Weather Condition | NimbusGS | Prev. SOTA | Gain |
|---------|----------|---------|------|
| Fog/Haze | SOTA | DehazeGS | Significant |
| Rain | SOTA | DeRainGS | Significant |
| Snow | SOTA | WeatherGS | Significant |
| Hybrid Weather | **New Benchmark** | No comparable method | — |

NimbusGS comprehensively outperforms all specialized methods under both individual and hybrid weather conditions.

### Ablation Study

| Configuration | PSNR | Note |
|------|------|------|
| 3DGS baseline only | Low | Weather severely degrades reconstruction |
| + Continuous scattering field | Improved | Global degradation removed |
| + Particulate residual layer | Further improved | Local interference removed |
| + Gradient scaling | Best | Distant geometry improved |

### Key Findings

- Physically driven decomposition is more robust than data-driven end-to-end methods, with a particularly pronounced advantage under hybrid weather conditions.
- Gradient scaling makes a critical contribution to reconstruction quality in distant and severely degraded regions.
- The unified framework generalizes without any modification to new weather types.

## Highlights & Insights

- **Physically Motivated Elegant Decomposition**: Weather is categorized into continuous and discrete types according to physical mechanisms, with each type handled by a corresponding modeling approach—physically principled and engineering-efficient.
- **Generality of Gradient Scaling**: This visibility-based adaptive optimization strategy is transferable to other degradation scenarios (e.g., low-light, underwater).
- **Value of a Unified Framework**: A single model handles all weather conditions, eliminating the engineering overhead of maintaining multiple specialized models.

## Limitations & Future Work

- The capacity of the particulate residual layer may be insufficient for extreme precipitation (e.g., heavy rainstorms).
- The continuous scattering field assumes a homogeneous atmosphere, limiting flexibility for non-uniform fog (e.g., localized dense fog).
- Performance under extreme weather combinations (e.g., snow + fog + rain) has not been validated.
- Future work may extend the framework to dynamic scenes (e.g., moving vehicles under adverse weather).

## Related Work & Insights

- **vs WeatherGS**: WeatherGS separates particles and lens artifacts but relies on 2D priors; NimbusGS performs unified modeling in 3D space.
- **vs DehazeNeRF/ScatterNeRF**: These methods address only fog/haze, whereas NimbusGS handles fog, rain, and snow simultaneously.
- **vs RainyScape/DeRainGS**: These rain-specific methods lack generalization to other weather types.

## Rating

- Novelty: ⭐⭐⭐⭐ Physical decomposition is conceptually clear; gradient scaling design is practically useful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across multiple weather conditions.
- Writing Quality: ⭐⭐⭐⭐ Physical motivation is articulated clearly.
- Value: ⭐⭐⭐⭐ Directly applicable to outdoor 3D reconstruction tasks such as autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](../../ICCV2025/3d_vision/robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [\[CVPR 2026\] Sky2Ground: A Benchmark for Site Modeling under Varying Altitude](sky2ground_a_benchmark_for_site_modeling_under_varying_altitude.md)
- [\[CVPR 2026\] FreeArtGS: Articulated Gaussian Splatting Under Free-Moving Scenario](freeartgs_articulated_gaussian_splatting_under_free-moving_scenario.md)
- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[CVPR 2026\] PoseMaster: A Unified 3D Native Framework for Stylized Pose Generation](posemaster_a_unified_3d_native_framework_for_stylized_pose_generation.md)

</div>

<!-- RELATED:END -->
