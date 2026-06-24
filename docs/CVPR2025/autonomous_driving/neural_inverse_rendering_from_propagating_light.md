---
title: >-
  [Paper Note] Neural Inverse Rendering from Propagating Light
description: >-
  [CVPR 2025][Autonomous Driving][Time-Resolved Inverse Rendering] The first method for physics-based inverse rendering from multi-view time-resolved LiDAR measurements (time-of-flight photon detection). It replaces recursive path tracing with a time-resolved radiance cache to model direct and indirect light transport, reducing normal MAE on synthetic scenes from 22.80° (FWP++) to 8.45°, while supporting novel view synthesis and relighting.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Time-Resolved Inverse Rendering"
  - "LiDAR"
  - "Indirect Illumination"
  - "Radiance Caching"
  - "Physics-Based Rendering"
date: 2026-05-08
content_hash: edd88afec54eee70
---

# Neural Inverse Rendering from Propagating Light

**Conference**: CVPR 2025  
**arXiv**: [2506.05347](https://arxiv.org/abs/2506.05347)  
**Code**: [https://anaghmalik.com/InvProp](https://anaghmalik.com/InvProp)  
**Area**: Autonomous Driving  
**Keywords**: Time-Resolved Inverse Rendering, LiDAR, Indirect Illumination, Radiance Caching, Physics-Based Rendering

## TL;DR

The first method for physics-based inverse rendering from multi-view time-resolved LiDAR measurements (time-of-flight photon detection). It replaces recursive path tracing with a time-resolved radiance cache to model direct and indirect light transport, reducing normal MAE on synthetic scenes from 22.80° (FWP++) to 8.45°, while supporting novel view synthesis and relighting.

## Background & Motivation

1. **Background**: Traditional inverse rendering methods recover geometry and materials from RGB images, but fail to handle strong indirect illumination (such as multiple bounces indoors) effectively. Time-resolved LiDAR (SPAD detectors) captures photon flight time, providing additional temporal constraints.
2. **Limitations of Prior Work**: (1) T-NeRF only models direct light, leading to severe distortion in scenes with strong indirect illumination; (2) FWP++ handles indirect light but lacks a physical model, restricting geometry reconstruction accuracy; (3) Recursive path tracing is computationally prohibitive and unsuitable for embedding in neural network optimization loops.
3. **Key Challenge**: Accurately modeling indirect light requires solving the complete rendering equation (recursive solving), but recursion is non-differentiable or computationally explosive.
4. **Goal**: Replace recursive path tracing with neural radiance caching to achieve differentiable physics-based inverse rendering.
5. **Key Insight**: The photon travel time in time-resolved data provides constraints on light path lengths—determining not only how much light reaches the detector but also how long the path taken is (distinguishing direct and indirect light).
6. **Core Idea**: Direct/indirect light decomposition + neural radiance caching (hash encoding) + split-sum approximation to handle indirect light BRDF integration.

## Method

### Overall Architecture

Multi-view time-resolved LiDAR measurements $\to$ Neural Geometry Network (density $\sigma$ + normal $n$) $\to$ Appearance Feature Hash Encoding $\to$ Direct Light Cache (analytic Fresnel + light source visibility) + Indirect Light Cache (split-sum approximation) $\to$ Disney-GGX BRDF $\to$ Volume Rendering and Transient Signal Comparison Optimization.

### Key Designs

1. **Time-Resolved Radiance Caching**

    - **Function**: Replaces recursive path tracing to efficiently model indirect light.
    - **Mechanism**: Splits incoming radiance into two terms: direct $L_o^{cache,dir}$ and indirect $L_o^{cache,indir}$. Direct light is computed with analytic BRDF + light source positions. Indirect light is approximated using split-sum: $L_o^{indir} = f_\Omega^{indir}(f^{app}, n, \omega_o) \cdot L_{i,\Omega}^{indir}(f^{app}, x_\ell, n, \omega_o)$, where both terms are approximated by MLPs.
    - **Design Motivation**: Radiance caching avoids recursive solving—it stores incoming radiance as a continuous function of space, direction, and time, requiring only table lookups.

2. **Direct/Indirect Light Decomposition**

    - **Function**: Accurately models direct light using physical BRDF models.
    - **Mechanism**: Direct light is calculated using the full Disney-GGX BRDF: $L_o^{dir} = f^{dir}(f^{app}, n, \omega_\ell, \omega_o) L_i^{dir}(x', \omega_\ell, \tau)(n \cdot \omega_\ell)$, while indirect light is approximated via split-sum due to analytical integration limits.
    - **Design Motivation**: Direct light has an analytical form (known light source positions), making exact modeling superior to approximation; indirect light integration is complex and can only be approximated.

3. **Multi-Resolution Hash Encoding**

    - **Function**: Efficiently represents spatially-varying appearance features.
    - **Mechanism**: $f^{app} = \mathcal{H}^{app}(x)$, leveraging multi-level hash encoding to capture material changes across different scales.
    - **Design Motivation**: Hash encoding has been shown in Instant-NGP to be highly efficient for spatial features.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{data} + \lambda_{cache} \mathcal{L}_{cache} + \lambda_{dir} \mathcal{L}_{dir} + \lambda_{indir} \mathcal{L}_{indir} + \text{regularizers}$. Normal smoothness, depth distortion, and mask regularizations are applied.

## Key Experimental Results

### Main Results

| Method | Synthetic PSNR↑ | Synthetic Normal MAE↓ | Synthetic Depth L1↓ |
|------|-----------|-------------|------------|
| T-NeRF | 22.44 | 28.00° | 0.59 |
| FWP++ | 29.00 | 22.80° | 0.47 |
| **Ours** | **30.99** | **8.45°** | **0.21** |

### Ablation Study

| Setting | Key Observations |
|------|---------|
| Direct light only | Severe distortion in scenes with strong indirect light |
| w/o split-sum | Degraded indirect light modeling |
| w/o time-resolved | Loss of light path length constraints |

### Key Findings

- Normal accuracy improved by 3.2x compared to FWP++ (22.80° $\to$ 8.45° MAE)—a key advantage of the physical model.
- On real captured data, PSNR is slightly lower than FWP++ (27.39 vs 28.45), possibly due to calibration errors.
- Supports relighting (re-rendering after changing light source positions), which FWP++ cannot do.

## Highlights & Insights

- **First combination of time-resolved LiDAR with physics-based inverse rendering**: Photon travel time provides constraints unattainable via conventional RGB.
- **Radiance caching replaces path tracing**: Converted the non-differentiable recursive process into a differentiable neural network lookup, which is highly elegant in engineering.
- **Supports relighting**: Obtains physical material parameters (albedo, roughness, metallic) and enables re-rendering under arbitrary illumination conditions.

## Limitations & Future Work

- Requires specialized hardware (picosecond lasers + SPAD detectors), making it unsuitable for consumer-grade devices.
- Calibration errors in real data directly affect reconstruction quality.
- The Disney-GGX BRDF cannot model all materials (e.g., transparency, subsurface scattering).
- Relighting requires fine-tuning direct/indirect loss, which is not fully automated.

## Related Work & Insights

- **vs T-NeRF**: Models only direct light, failing completely in the Cornell Box scene with strong indirect illumination.
- **vs FWP++**: Non-physical model leading to poor geometric accuracy (normal MAE 22.8°), and does not support relighting.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First combination of time-resolved LiDAR and physics-based inverse rendering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + real data + multiple metrics, but limited number of scenes.
- Writing Quality: ⭐⭐⭐⭐ Rigorous physical derivations.
- Value: ⭐⭐⭐⭐ Opens up new directions for inverse rendering in indirect illumination scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LightLoc: Learning Outdoor LiDAR Localization at Light Speed](lightloc_learning_outdoor_lidar_localization_at_light_speed.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](../../ICCV2025/autonomous_driving/gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[CVPR 2025\] Single Pixel Image Classification using an Ultrafast Digital Light Projector](single_pixel_image_classification_using_an_ultrafast_digital_light_projector.md)
- [\[ICML 2025\] GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction](../../ICML2025/autonomous_driving/goirl_graph-oriented_inverse_reinforcement_learning_for_multimodal_trajectory_pr.md)

</div>

<!-- RELATED:END -->
