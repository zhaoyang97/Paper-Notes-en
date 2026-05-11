---
title: >-
  [Paper Note] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering
description: >-
  [CVPR 2026][3D Vision][inverse rendering] SGS-Intrinsic proposes a two-stage indoor inverse rendering framework. Stage I constructs a geometrically consistent dense Gaussian field guided by semantic and geometric priors.…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "inverse rendering"
  - "sparse-view"
  - "Gaussian splatting"
  - "material decomposition"
  - "indoor scenes"
date: 2026-05-08
content_hash: cee93ad76b1b943f
---

# SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering

**Conference**: CVPR 2026
**arXiv**: [2603.27516](https://arxiv.org/abs/2603.27516)
**Code**: [https://github.com/GrumpySloths/SGS_Intrinsic.github.io](https://github.com/GrumpySloths/SGS_Intrinsic.github.io)
**Area**: 3D Vision
**Keywords**: inverse rendering, sparse-view, Gaussian splatting, material decomposition, indoor scenes

## TL;DR

SGS-Intrinsic proposes a two-stage indoor inverse rendering framework. Stage I constructs a geometrically consistent dense Gaussian field guided by semantic and geometric priors. Stage II performs material–illumination decomposition via a hybrid lighting model and material priors, with a dedicated de-shadowing module to prevent shadow baking into albedo.

## Background & Motivation

Sparse-view indoor inverse rendering is a severely ill-posed problem: supervision signals are scarce, indoor illumination is complex (near-field and high-frequency), and material and lighting are strongly coupled. Existing methods either focus solely on geometric reconstruction without material decomposition, assume distant light sources (unsuitable for indoor scenes), or fail under sparse-view conditions.

**Three core challenges**: (1) unreliable Gaussian geometry reconstruction under sparse views; (2) difficulty modeling near-field high-frequency indoor illumination; (3) cast shadows being incorrectly baked into material appearance.

## Method

### Overall Architecture

Two stages: Stage I initializes a dense point cloud via VGGT and builds a high-quality Gaussian geometry field supervised by normal and semantic priors; Stage II performs inverse rendering on this foundation using a hybrid lighting model (environment map + spherical Gaussian mixture), diffusion-based material priors, and a de-shadowing module.

### Key Designs

1. **Prior-Guided Dense Geometry Reconstruction**:

    - Function: Establish reliable Gaussian geometric foundations under sparse views.
    - Mechanism: VGGT replaces traditional SfM to obtain dense scene layout point clouds. StableNormal provides normal supervision $\mathcal{L}_{normal} = 1 - \hat{n}^T n_m$, and LSEG provides semantic supervision. A semantic consistency constraint between training views and virtual views is additionally introduced to prevent overfitting.
    - Design Motivation: Traditional SfM yields only sparse point clouds under sparse views, insufficient to support Gaussian optimization. Dense priors from pretrained models compensate for the lack of supervision.

2. **Hybrid Lighting Model + Material Priors**:

    - Function: Accurately model complex indoor illumination and enable effective material–lighting decomposition.
    - Mechanism: An environment map captures distant ambient light, while a spherical Gaussian mixture (SGM) models near-field high-frequency illumination. Diffusion model-based material priors enforce cross-view and cross-illumination material consistency, yielding illumination- and view-invariant material reconstruction.
    - Design Motivation: A single lighting model lacks flexibility; the hybrid scheme handles different frequency components separately. Material priors help resolve the inherent material–lighting ambiguity.

3. **Lightweight De-shadowing Module**:

    - Function: Prevent cast shadows from being incorrectly baked into albedo.
    - Mechanism: A lightweight de-shadowing model explicitly models visibility, attributing dark regions to occlusion rather than material properties. Combined with illumination-invariant material consistency constraints, this ensures the same material produces consistent albedo under varying lighting conditions.
    - Design Motivation: Indoor scenes contain complex shadows; without explicit handling, the optimizer tends to absorb shadow effects into material color.

### Loss & Training

Stage I: RGB reconstruction loss + normal loss + semantic consistency loss. Stage II: PBR rendering loss + material consistency loss + de-shadowing regularization.

## Key Experimental Results

### Main Results

| Method | Interiorverse NVS PSNR | Albedo Accuracy | Note |
|--------|------------------------|-----------------|------|
| GeoSplat | Low | Low | Insufficient geometry |
| IRGS | Medium | Medium | Limited lighting model |
| **SGS-Intrinsic** | **Best** | **Best** | Outperforms across all metrics |

The proposed method achieves state-of-the-art performance on novel view synthesis and inverse rendering metrics across benchmark datasets.

### Ablation Study

| Configuration | NVS Quality | Material Decomposition | Note |
|---------------|-------------|------------------------|------|
| w/o prior guidance | Significant drop | Poor | Unreliable geometry degrades downstream |
| w/o hybrid lighting | — | Degraded | Insufficient near-field modeling |
| w/o de-shadowing | — | Shadow baking | Albedo contaminated by shadows |
| Full model | Best | Best | All components necessary |

### Key Findings

- Dense initialization via VGGT is the critical foundation for success under sparse views—high-quality geometry is a prerequisite for high-quality inverse rendering.
- The de-shadowing module yields substantial improvement in albedo estimation quality; shadow baking is the dominant source of material estimation error in indoor scenes.
- Semantic consistency constraints effectively prevent overfitting under sparse-view conditions.

## Highlights & Insights

- **Rationale for two-stage decoupling**: Geometry and material decomposition have a clear dependency—stabilizing geometry before decomposing materials is more robust than end-to-end joint optimization.
- **De-shadowing as an explicit module**: Modeling shadows explicitly rather than leaving them for the optimizer to handle implicitly is a simple yet critical design choice.
- **Pretrained models as prior sources**: The combined use of StableNormal, LSEG, VGGT, and diffusion models demonstrates an effective strategy for compensating data scarcity with rich priors in sparse-view settings.

## Limitations & Future Work

- Reliance on multiple pretrained models (VGGT, StableNormal, LSEG, diffusion model) results in high system complexity.
- Limited capacity to handle non-Lambertian materials such as specular or transparent surfaces.
- Two-stage training is less efficient than end-to-end approaches.
- Future work may explore reducing dependence on prior models or unifying them into a single model.

## Related Work & Insights

- **vs. GeoSplat/IRGS**: Both are 3DGS-based inverse rendering methods; SGS-Intrinsic achieves superior results through stronger priors and explicit de-shadowing.
- **vs. NeRF-based inverse rendering**: The explicit representation of 3DGS enables more straightforward disentanglement of PBR attributes.
- **vs. single-image inverse rendering**: Multi-view methods inherently provide 3D consistency, though sparse views introduce additional challenges.

## Rating

- Novelty: ⭐⭐⭐⭐ Each module is well-designed and the de-shadowing idea is valuable, though the overall contribution is largely a combination of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive benchmark comparisons and clear ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method description is systematic and clear.
- Value: ⭐⭐⭐⭐ Direct practical value for indoor AR/VR applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RadioGS: Radiometrically Consistent Gaussian Surfels for Inverse Rendering](../../ICLR2026/3d_vision/radiogs_radiometric_gaussian_surfels.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](../../ICCV2025/3d_vision/geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](../../AAAI2026/3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[CVPR 2026\] LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates](ltgs_long-term_gaussian_scene_chronology_from_sparse_view_updates.md)

</div>

<!-- RELATED:END -->
