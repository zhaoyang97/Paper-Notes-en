---
title: >-
  [Paper Note] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes MuGS, the first generalizable 3D Gaussian Splatting method designed for multi-baseline settings. By fusing multi-view stereo (MVS) and monocular depth estimation (MDE) features and introducing a projected-sampled depth consistency network, MuGS achieves state-of-the-art novel view synthesis under both small-baseline and large-baseline scenarios.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Multi-Baseline Generalization
  - Monocular Depth Estimation
  - Multi-View Stereo
  - Novel View Synthesis
date: 2026-05-08
content_hash: 0c931b17ec7c06f7
---

# MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2508.04297](https://arxiv.org/abs/2508.04297)
**Code**: [https://github.com/EuclidLou/MuGS](https://github.com/EuclidLou/MuGS)
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: 3D Gaussian Splatting, Multi-Baseline Generalization, Monocular Depth Estimation, Multi-View Stereo, Novel View Synthesis

## TL;DR
This paper proposes MuGS, the first generalizable 3D Gaussian Splatting method designed for multi-baseline settings. By fusing multi-view stereo (MVS) and monocular depth estimation (MDE) features and introducing a projected-sampled depth consistency network, MuGS achieves state-of-the-art novel view synthesis under both small-baseline and large-baseline scenarios.

## Background & Motivation

**State of the Field**: 3D Gaussian Splatting (3D-GS) has become the dominant paradigm for novel view synthesis due to its efficient real-time rendering, but it requires per-scene optimization. Generalizable methods enable feed-forward inference on unseen scenes via data-driven learning; however, existing approaches are specialized either for small-baseline (high image overlap) or large-baseline (low image overlap) settings.

**Limitations of Prior Work**: Small-baseline methods suffer from large depth errors on large-baseline data due to occlusions and insufficient overlap; large-baseline methods produce inaccurate depth estimates on small-baseline data due to the lack of matching cues. The core bottleneck lies in the fact that the optimal depth estimation strategy varies with the baseline.

**Root Cause**: Small-baseline settings rely on MVS matching, while large-baseline settings rely on monocular geometric priors. These two strategies are fundamentally different and difficult to unify within a single model.

**Paper Goals**: Construct the first generalizable Gaussian Splatting method capable of handling multiple baseline configurations simultaneously.

**Starting Point**: Accurate depth guidance can unify the challenges posed by both baseline settings. MVS yields high accuracy when overlap is sufficient but fails in challenging regions, while MDE provides more robust and smooth depth maps but lacks multi-view consistency. The two are complementary.

**Core Idea**: Fuse the projected depth from MVS matching and the sampled depth from MDE predictions, and use a 3D U-Net to compute their consistency for constructing a refined probability volume that guides depth regression.

## Method

### Overall Architecture
Given sparse multi-view images as input, the method extracts image features and obtains auxiliary depth maps using a pretrained monocular depth model. During MVS cost volume construction, consistency between projected depth and sampled depth is computed simultaneously. A 3D U-Net refines the probability volume, from which depth and Gaussian parameters are regressed for novel view rendering.

### Key Designs

1. **Projected-Sampled Depth Consistency Network**:

    - **Function**: Fuses MVS matching information with monocular depth priors.
    - **Mechanism**: For each depth hypothesis, two types of depth are computed: projected depth (spatial position derived from multi-view feature warping) and sampled depth (expected depth from the monocular depth model). A 3D U-Net computes a consistency score between the two, which is used to refine the cost volume. Depth hypotheses with higher consistency receive higher probabilities.
    - **Design Motivation**: MVS is accurate in overlapping regions but fails under occlusion, while MDE is robust everywhere but suffers from scale ambiguity. The consistency network adaptively selects reliable information from each source.

2. **Refined Probability Volume Guidance**:

    - **Function**: Converts consistency information into a guidance signal for depth regression.
    - **Mechanism**: Consistency scores serve as queries in an attention network, which refines the depth probability volume through a lightweight attention mechanism, concentrating probability mass near the true surface. An MLP network then regresses Gaussian parameters using features and colors sampled from each source view.
    - **Design Motivation**: By prioritizing depth hypotheses near the surface, the approach reduces the influence of noisy features sampled at incorrect depths on rendering quality.

3. **Reference View Loss**:

    - **Function**: Provides contextual supervision to improve geometric consistency.
    - **Mechanism**: In addition to supervising rendered target views, the method additionally renders images from reference views (input views) and computes loss against the original images, leveraging the precise correspondences of known viewpoints to more effectively learn geometry.
    - **Design Motivation**: Under sparse input, supervision from target views is limited; the reference view loss provides additional geometric constraints.

### Loss & Training
Rendering loss (L1 + SSIM) is applied to both target views and reference views. A 3D Gaussian representation is used to accelerate training and inference.

## Key Experimental Results

### Main Results

| Dataset | Baseline Type | Ours PSNR | Prev. SOTA PSNR | Gain |
|--------|---------|-----------|-------------|------|
| DTU (3-view) | Small-baseline | SOTA | 2nd best | Significant |
| RealEstate10K | Large-baseline | SOTA | 2nd best | Significant |
| LLFF (zero-shot) | Small-baseline | Competitive | - | Generalization |
| Mip-NeRF 360 (zero-shot) | Large-baseline | Competitive | - | Generalization |

### Ablation Study

| Configuration | Performance | Notes |
|------|------|------|
| MVS only | Baseline | Large depth errors under large baselines |
| + MDE features | Improved | Monocular depth as auxiliary |
| + Depth consistency network | Further improved | Intelligent fusion of both depth cues |
| + Reference view loss | Best | Additional geometric supervision |

### Key Findings
- First generalizable Gaussian method to simultaneously achieve SOTA on both small-baseline and large-baseline settings.
- The depth consistency network is the core contribution — removing it leads to significant performance drops under both baseline types.
- Strong zero-shot generalization demonstrates that the method learns universal multi-baseline depth reasoning capabilities.

## Highlights & Insights
- **Complementary MVS + MDE Fusion**: Rather than naively concatenating features, a consistency network intelligently selects reliable information sources. This design is transferable to other tasks requiring fusion of multiple depth estimation modalities.
- **Unified Multi-Baseline Handling**: The first work to address the baseline generalization problem in generalizable Gaussian Splatting, offering strong practical value.

## Limitations & Future Work
- Requires a pretrained monocular depth model as auxiliary input, increasing system complexity.
- Performance remains limited under extreme large-baseline settings (complete absence of overlap).
- Adaptive baseline detection to select different fusion strategies is a promising future direction.

## Related Work & Insights
- **vs. MuRF**: MuRF relies on MVS density volumes, where density becomes diffuse under occlusion; this work addresses the problem from the perspective of depth accuracy in a unified manner.
- **vs. DepthSplat**: DepthSplat fuses MVS and MDE via feature-level concatenation, whereas this work explicitly models the relationship between the two depth cues.

## Rating
- Novelty: ⭐⭐⭐⭐ Multi-baseline generalization is a novel problem; the consistency network design is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation with zero-shot validation.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ High practical value; multi-baseline scenarios are common in real-world applications.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgs-lm_faster_gaussian-splatting_optimization_with_levenberg-marquardt.md)

<!-- RELATED:END -->
