---
title: >-
  [Paper Note] PD²GS: Part-Level Decoupling and Continuous Deformation of Articulated Objects via Gaussian Splatting
description: >-
  [ICLR 2026][3D Vision][articulated objects] PD²GS is proposed as a framework that learns a shared canonical Gaussian field and models each interaction state as a continuous deformation thereof…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "articulated objects"
  - "3D Gaussian Splatting"
  - "part segmentation"
  - "continuous deformation"
  - "SAM"
date: 2026-05-08
content_hash: a033cd31dc0be1b8
---

# PD²GS: Part-Level Decoupling and Continuous Deformation of Articulated Objects via Gaussian Splatting

**Conference**: ICLR 2026
**arXiv**: [2506.09663](https://arxiv.org/abs/2506.09663)  
**Code**: Available  
**Area**: 3D Vision / Articulated Object Modeling
**Keywords**: articulated objects, 3D Gaussian Splatting, part segmentation, continuous deformation, SAM

## TL;DR
PD²GS is proposed as a framework that learns a shared canonical Gaussian field and models each interaction state as a continuous deformation thereof, enabling part-level decoupling, reconstruction, and continuous control of articulated objects via coarse-to-fine motion trajectory clustering and SAM-guided boundary refinement, without any manual supervision.

## Background & Motivation

**Background**: 3D modeling of articulated objects (doors, drawers, laptops) is critical for robotics, AR/VR, and digital twins. Recent works such as PARIS and GAPartNet employ NeRF/3DGS for self-supervised modeling, yet are largely limited to single-joint, two-state settings.

**Limitations of Prior Work**: (1) Two-state methods support only discrete pairwise comparisons and cannot model continuous motion; (2) they require prior knowledge of the number of parts or strict geometric constraints; (3) multi-part decoupling relies on Marching Cubes explicit meshes, leading to severe error accumulation.

**Key Challenge**: How can a continuous part-level motion model be learned from limited discrete interaction-state observations?

**Goal**: Self-supervised learning from multi-view, multi-state images to achieve (1) part-aware reconstruction, (2) part-level continuous control, and (3) accurate kinematic modeling.

**Key Insight**: The key insight is that each interaction state can be modeled as a continuous deformation of a shared canonical Gaussian field, where motion within a part is consistent while motion across parts differs.

**Core Idea**: A latent-code-conditioned deformation network drives continuous deformation of the canonical Gaussian field; automatic part decoupling is achieved through motion trajectory clustering followed by SAM-guided boundary refinement.

## Method

### Overall Architecture
The input consists of multi-view images of an articulated object captured across $K$ interaction states. The pipeline comprises: (1) construction of a canonical Gaussian field with a latent-code-conditioned deformation MLP; (2) coarse-grained part clustering based on motion trajectories; (3) SAM-guided boundary refinement; and (4) kinematic analysis and continuous control.

### Key Designs

1. **Deformable Gaussian Splatting**:

    - Function: Unifies discrete interaction states as continuous deformations of a shared canonical field.
    - Mechanism: Each state $k$ is associated with a latent code $\alpha_k \in \mathbb{R}^D$; an MLP $f_{def}$ predicts per-Gaussian displacements $(\Delta\mu_i, \Delta q_i, \Delta s_i) = f_{def}(\mu_i, q_i, s_i | \alpha_k)$, applied via addition (position) and quaternion multiplication (rotation).
    - Design Motivation: Latent code parameterization enables interpolation to unseen states after training.

2. **Coarse Motion-Driven Part Segmentation**:

    - Function: Automatically discovers parts from motion trajectories.
    - Mechanism: (a) The maximum displacement of each Gaussian across $K$ states is computed, and a threshold separates static from dynamic regions; (b) a VLM (BLIP/Gemini) estimates the number of moving parts from image pairs via majority voting; (c) motion descriptors (normalized direction + displacement magnitude) are constructed and clustered on the unit sphere via K-means.
    - Design Motivation: Gaussians belonging to the same rigid part share consistent motion directions; after normalization, their angular distances remain small even when magnitudes differ.

3. **SAM-Guided Boundary Refinement**:

    - Function: Refines part boundaries.
    - Mechanism: 3D-to-2D projected prompt points are generated for Gaussians near part boundaries; SAM is invoked to produce 2D masks, which are back-projected into 3D to correct part labels. Boundary Gaussians are further split into multiple smaller Gaussians with reassigned labels.
    - Design Motivation: Motion clustering yields coarse boundaries, while the visual prior from SAM provides pixel-accurate segmentation.

### Loss & Training
$$\mathcal{L}_{total} = \mathcal{L}_{photo} + \mathcal{L}_{D_{SIMM}}$$
comprising a photometric reconstruction loss and a density similarity regularization term.

## Key Experimental Results

### Main Results (PartNet-Mobility)

| Method | PSNR↑ | SSIM↑ | Part IoU↑ | Joint Error↓ |
|--------|-------|-------|-----------|-------------|
| PARIS | Low | Low | ~50% | High |
| CAGE | Medium | Medium | ~55% | Medium |
| **PD²GS** | **Highest** | **Highest** | **~70%** | **Lowest** |

### Ablation Study

| Configuration | Reconstruction Quality | Segmentation Accuracy | Notes |
|---------------|----------------------|-----------------------|-------|
| Full PD²GS | Best | Best | Complete model |
| w/o SAM refinement | Slightly lower | −~10% | Coarse clustering boundaries imprecise |
| w/o VLM counting | Comparable | −~5% | Manually specified $K$ yields similar results |
| 2 states vs. 4 states | Lower | Lower | More states provide stronger motion constraints |

### Key Findings
- **Continuous control**: Interpolating latent codes generates smooth intermediate states, whereas prior methods can only jump between discrete states.
- **Multi-part support**: Complex objects with multiple independently moving parts, such as multi-drawer cabinets, are successfully handled.
- **RS-Art real data**: Strong performance on the authors' real-to-simulation dataset validates sim-to-real generalization.

## Highlights & Insights
- **Elegant latent-code-driven continuous deformation**: Discrete state observations are encoded into a continuous motion space, enabling generation of unseen configurations.
- **Self-supervised motion-as-segmentation paradigm**: Part structure is automatically discovered from motion differences without any manual annotation.
- **Novel 3D-to-2D SAM prompting**: 2D prompt points are automatically derived from 3D part boundaries, eliminating the need for manually annotated SAM prompts.

## Limitations & Future Work
- VLM-based part counting can be unreliable; majority voting is required for stability.
- The motion threshold $\tau_{mot}$ requires manual tuning.
- Only rigid articulated motion is supported; flexible deformations (e.g., cloth) are not addressed.
- The RS-Art dataset remains limited in scale.

## Related Work & Insights
- **vs. PARIS**: PARIS supports only single-joint two-state settings, whereas PD²GS enables multi-part, multi-state continuous control.
- **vs. dynamic 3DGS (4D-GS, etc.)**: Dynamic methods do not distinguish part-level motion semantics; PD²GS performs explicit decoupling.
- The framework has direct applicability to robotic object manipulation, as inferred part identities and kinematic parameters can directly inform manipulation planning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of canonical field + latent deformation + automatic part discovery is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers PartNet-Mobility and RS-Art real data with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with well-formatted formulations.
- Value: ⭐⭐⭐⭐⭐ A significant advance in continuous modeling of articulated objects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation](learning_part-aware_dense_3d_feature_field_for_generalizable_articulated_object_.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](../../CVPR2026/3d_vision/extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](../../CVPR2026/3d_vision/ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](../../AAAI2026/3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[ICLR 2026\] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data](partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)

</div>

<!-- RELATED:END -->
