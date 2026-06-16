---
title: >-
  [Paper Note] Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting
description: >-
  [NeurIPS 2025][Autonomous Driving][Gaussian Splatting] A multi-scale bilateral grid pyramid is proposed to unify global appearance codes and pixel-level bilateral grids. A 3-level hierarchy (coarse→medium→fine) captures…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Gaussian Splatting"
  - "Bilateral Grid"
  - "Appearance Modeling"
  - "Driving Scene"
  - "Photometric Consistency"
date: 2026-05-08
content_hash: 68d071d9b110d3c2
---

# Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2506.05280](https://arxiv.org/abs/2506.05280)  
**Code**: [https://bigcileng.github.io/bilateral-driving](https://bigcileng.github.io/bilateral-driving)  
**Area**: Autonomous Driving / Neural Rendering
**Keywords**: Gaussian Splatting, Bilateral Grid, Appearance Modeling, Driving Scene, Photometric Consistency

## TL;DR
A multi-scale bilateral grid pyramid is proposed to unify global appearance codes and pixel-level bilateral grids. A 3-level hierarchy (coarse→medium→fine) captures global/regional/pixel-level photometric variation respectively. By employing a luminance-guided slice-and-blend pipeline and adaptive regularization, the method addresses photometric inconsistency in driving scene 3DGS, achieving a 28.2% improvement in Chamfer Distance over OmniRe on Waymo.

## Background & Motivation

**Background**: 3D Gaussian Splatting reconstruction of driving scenes suffers from cross-image photometric inconsistency caused by auto-exposure and white balance variation. Existing methods compensate via either global appearance codes (one vector per image) or pixel-level bilateral grids.

**Limitations of Prior Work**: Global appearance codes have limited expressiveness, modeling only overall color tone shifts. Pixel-level bilateral grids have large parameter counts (27.8M) and are difficult to optimize—prone to overfitting on small-scale driving datasets. Each approach has distinct advantages but they are incompatible with one another.

**Key Challenge**: The model must simultaneously capture global tone shifts (where appearance codes excel) and local illumination variation (where bilateral grids excel), yet the two employ fundamentally different parameterizations.

**Goal**: Design a unified framework that achieves both global and local photometric compensation with a controllable parameter budget.

**Key Insight**: A multi-scale grid pyramid—coarse grid ≈ global appearance code, medium grid ≈ regional compensation, fine grid ≈ pixel-level adjustment. The three levels are progressively stacked and composed sequentially.

**Core Idea**: A 3-level bilateral grid pyramid ($2{\times}2{\times}1 \to 4{\times}4{\times}2 \to 8{\times}8{\times}4$) unifies global/regional/pixel-level photometric modeling, coupled with luminance-guided slicing and adaptive regularization.

## Method

### Overall Architecture
3DGS renders image $I^r$ → luminance guidance map extraction → multi-scale bilateral grid pyramid (3 levels) each sliced to obtain affine transforms $\mathcal{T}^{(l)}$ → sequential composition $I^e = \mathcal{T}^{(2)} \circ \mathcal{T}^{(1)} \circ \mathcal{T}^{(0)}(I^r)$ → reconstruction loss + regularization

### Key Designs

1. **Multi-Scale Bilateral Grid Pyramid**:

    - Function: Models photometric variation at different spatial resolutions.
    - Mechanism: Level 0 ($2{\times}2{\times}1{\times}12$) = global appearance code—extremely low spatial resolution with only 4 spatial positions per level; Level 1 ($4{\times}4{\times}2{\times}12$) = regional compensation—captures regional shadows and highlights; Level 2 ($8{\times}8{\times}4{\times}12$) = pixel-level refinement—handles fine illumination boundaries. Each level outputs 12-channel affine transform coefficients ($3{\times}4$ matrix for RGB).
    - Design Motivation: Coarse grids provide stable global compensation (analogous to appearance codes), while fine grids offer flexible local adjustment (analogous to bilateral grids), yet the total parameter count is only 3.97M (vs. 27.8M for a single bilateral grid).

2. **Luminance-Guided Slice-and-Blend**:

    - Function: Uses luminance information from the rendered image to guide bilateral grid queries.
    - Mechanism: A luminance guidance map is extracted from the rendered image and used to query the corresponding affine coefficients along the guidance dimension of the bilateral grid. Each level employs a guidance map with a different downsampling rate.
    - Design Motivation: Luminance is the primary indicator of photometric variation—bright and dark regions require different compensation strategies.

3. **Adaptive Regularization**:

    - Function: Prevents grid overfitting while preserving sufficient expressiveness.
    - Mechanism: Circle regularization constrains affine transforms to remain close to the identity; adaptive Total Variation (TV) regularization enforces spatial smoothness between neighboring grid cells.
    - Design Motivation: Without regularization, fine grids tend to produce artifacts (unrealistic local color changes).

### Loss & Training
- $\mathcal{L} = \mathcal{L}_{recon} + \lambda_{circle}\mathcal{L}_{circle} + \lambda_{TV}\mathcal{L}_{TV}$
- Training time: 2.10 hours (vs. longer training and 7× more parameters for single bilateral grid methods).

## Key Experimental Results

### Main Results

| Dataset | Method | Chamfer Distance↓ | PSNR↑ | SSIM↑ |
|--------|------|-------------------|-------|-------|
| Waymo | OmniRe | 1.482 | — | — |
| Waymo | OmniRe + Appearance Code | 1.378 | — | — |
| Waymo | **Ours** | **0.989** | — | — |
| NuScenes | **Ours** | **1.161** | 27.69 | 0.847 |
| Argoverse | **Ours** | **0.807** | — | — |

CD improvement on Waymo: **28.2%** (1.378→0.989)

### Ablation Study

| Configuration | NVS PSNR | CD |
|------|----------|-----|
| Coarse grid only (2,2,1) | 24.33 | 1.231 |
| Fine grid only (8,8,4) | — | 1.376 (worse) |
| **3-level pyramid** | **24.64** | **1.161** |
| w/o Circle regularization | 24.60 | — |
| w/o TV regularization | 24.56 | — |

### Key Findings
- Using the fine grid alone performs worse (CD 1.376 vs. coarse 1.231)—overfitting due to the lack of global constraints.
- The 3-level pyramid (3.97M parameters) uses far fewer parameters than a single bilateral grid (27.8M) while achieving superior results.
- Gains are larger on challenging scenes (18 high-photometric-variation scenes): +1.29 dB PSNR.
- Regularization is critical for stable training, with each term contributing ~0.03–0.04 dB individually.

## Highlights & Insights
- The **"coarse as global, fine as local" unification** is particularly elegant: transitioning naturally from global appearance codes to pixel-level grids simply by varying spatial resolution.
- **7× parameter efficiency**: 3.97M vs. 27.8M demonstrates that multi-scale decomposition is far more effective than naively scaling a single grid.
- **Sequential composition of transforms** ensures cross-scale cooperation—the coarse level handles large corrections first, while the fine level performs residual refinement.

## Limitations & Future Work
- Computational overhead is higher than simple appearance codes, requiring forward passes and regularization across 3 grid levels.
- LiDAR-camera misalignment caused by fast motion or non-rigid objects remains a challenge.
- Validation on nighttime or extreme weather conditions is absent—photometric variation patterns may differ substantially from daytime.
- Fixed spatial resolutions (2/4/8) in the bilateral grid may not generalize well to scenes of all scales.
- Evaluation is limited to the 3DGS framework; NeRF or other neural rendering backends are not tested.
- The luminance-based guidance dimension design may overlook photometric variation along the chrominance axes.

## Related Work & Insights
- **vs. OmniRe + Appearance Code**: Global codes can only compensate for tone shifts (CD 1.378); the proposed multi-scale pyramid achieves 0.989.
- **vs. 3DGS-DR (single bilateral grid)**: High parameter count and optimization difficulty; the multi-scale approach is more efficient and stable.
- **vs. StreetGS**: StreetGS relies on global appearance codes and cannot handle local illumination variation.
- **Transferability**: The multi-scale bilateral grid is applicable to any neural rendering scenario with photometric inconsistency (e.g., indoor scenes, day-night transitions).

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-scale unification idea is clean and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four driving datasets, challenging scene analysis, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear.
- Value: ⭐⭐⭐⭐ Addresses a key engineering problem in driving scene 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](../../ICCV2025/autonomous_driving/ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](../../ICCV2025/autonomous_driving/splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[ICCV 2025\] CoDa-4DGS: Dynamic Gaussian Splatting with Context and Deformation Awareness for Autonomous Driving](../../ICCV2025/autonomous_driving/coda-4dgs_dynamic_gaussian_splatting_with_context_and_deformation_awareness_for_.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[NeurIPS 2025\] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)

</div>

<!-- RELATED:END -->
