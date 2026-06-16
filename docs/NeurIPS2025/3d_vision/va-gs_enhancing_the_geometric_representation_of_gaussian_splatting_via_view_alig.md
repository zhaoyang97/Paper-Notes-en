---
title: >-
  [Paper Note] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] By introducing four view alignment strategies — edge-aware image supervision, visibility-aware multi-view photometric alignment, normal consistency constraints…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "surface reconstruction"
  - "multi-view alignment"
  - "normal consistency"
  - "geometric representation"
date: 2026-05-08
content_hash: dcebf2de4ddeeee0
---

# VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment

**Conference**: NeurIPS 2025
**arXiv**: [2510.11473](https://arxiv.org/abs/2510.11473)  
**Code**: [GitHub](https://github.com/LeoQLi/VA-GS)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, surface reconstruction, multi-view alignment, normal consistency, geometric representation

## TL;DR

By introducing four view alignment strategies — edge-aware image supervision, visibility-aware multi-view photometric alignment, normal consistency constraints, and depth image feature alignment — VA-GS significantly improves the geometric representation accuracy of 3D Gaussian Splatting, achieving state-of-the-art performance in surface reconstruction and novel view synthesis.

## Background & Motivation

3D Gaussian Splatting (3DGS) has rapidly become a mainstream scene representation method due to its real-time rendering and high-quality view synthesis capabilities. However, 3DGS still exhibits significant limitations in accurate surface reconstruction. The root cause lies in the **discrete and unstructured nature** of Gaussian primitives: optimizing solely with RGB rendering loss leads to insufficient geometric accuracy, particularly in two scenarios:

**Illumination-induced artifacts**: Shadows and specular highlights distort the photometric loss, causing geometric drift in the reconstruction.

**Boundary ambiguity**: The directional uncertainty of Gaussian primitives at object boundaries results in poorly defined surface edges.

Existing methods such as SuGaR extract meshes via density fields but are computationally expensive; 2DGS uses 2D surface elements but fails in unbounded scenes; GS-Pull introduces SDFs but produces overly smooth surfaces; PGSR adopts a local planar assumption but cannot handle illumination issues.

The core idea of this paper is to reformulate surface reconstruction as a **view alignment** problem — by enforcing edge and normal alignment within single views, and photometric and feature alignment across multiple views, the geometric properties of Gaussian primitives are constrained from multiple complementary perspectives, achieving high-fidelity surface reconstruction without introducing any additional implicit representation.

## Method

### Overall Architecture

Given a set of posed RGB images, the goal is to learn a collection of 3D Gaussian functions and their attributes (color, opacity, position, shape) to represent the 3D scene geometry. The method augments standard 3DGS with five loss functions: $\mathcal{L}_I$ (edge-aware image reconstruction), $\mathcal{L}_{nc}$ (normal consistency), $\mathcal{L}_{ns}$ (normal smoothness), $\mathcal{L}_p$ (multi-view photometric alignment), and $\mathcal{L}_f$ (multi-view feature alignment).

### Key Designs

1. **Edge-Aware Image Reconstruction ($\mathcal{L}_I$)**: Built upon the standard L1+D-SSIM loss, an image gradient term $\beta_2 L_1(\nabla\tilde{I} - \nabla I)$ is added to supervise edge information. The design motivation is that the original color loss over-smooths high-frequency regions; adding a gradient constraint preserves sharp structures and boundary details.

2. **Edge-Aware Normal Consistency ($\mathcal{L}_{nc}$)**: Aligns the Gaussian primitive normals $\tilde{N}$ with the depth-gradient normals $\hat{N}$, using an edge weight $\delta = (1-\nabla I)^2$ to reduce the loss contribution at edge regions. The design motivation is that Gaussian normal directions at edges are ambiguous, and forcing alignment there would introduce erroneous supervision; thus the loss is down-weighted at edges.

3. **Normal Smoothness Constraint ($\mathcal{L}_{ns}$)**: Penalizes normal differences between neighboring pixels, using a threshold $\tau$ and a ReLU gating mechanism to distinguish true geometric edges from noise. This addresses normal noise in texture-less regions and spurious edges introduced by illumination variation.

4. **Visibility-Aware Multi-View Photometric Alignment ($\mathcal{L}_p$)**: Drawing on classical MVS methods, reference view pixels are projected onto source views via homography $H_{rs}$, and NCC photometric consistency is computed. Key innovations include:

    - **Visibility term** $\upsilon_{rs}$: determines whether the projected point falls within the source view's field of view.
    - **Occlusion weight** $\omega$: filters occluded or geometrically erroneous pixels using the **reprojection error** $\varphi$, where $\omega = 1/\exp(\varphi)$ (when $\varphi < 1$).

5. **Multi-View Feature Alignment ($\mathcal{L}_f$)**: Extracts deep image features using a pretrained network and computes the cosine similarity between features at corresponding locations in the reference and source views. The design motivation is that image-level losses are sensitive to noise, blur, and illumination changes, whereas high-dimensional feature spaces offer greater robustness.

### Loss & Training

The final loss is: $\mathcal{L} = \mathcal{L}_I + \lambda_1\mathcal{L}_{nc} + \lambda_2\mathcal{L}_{ns} + \lambda_3\mathcal{L}_p + \lambda_4\mathcal{L}_f$

Training proceeds in stages:
- First 7,000 steps: pretraining with color loss only to obtain a coarse geometric initialization.
- Addition of edge terms and normal alignment.
- Addition of multi-view photometric alignment (8,000 steps).
- Addition of multi-view feature alignment (5,000 steps).
- Final retraining for novel view synthesis (10,000 steps).

## Key Experimental Results

### Main Results: DTU Surface Reconstruction (Chamfer Distance ↓)

| Method | Mean CD | Training Time |
|--------|---------|--------------|
| 3DGS | 1.96 | 3.4m |
| 2DGS | 0.80 | 5.8m |
| GS-Pull | 0.75 | 5.6m |
| PGSR | 0.53 | 15m |
| GausSurf | 0.52 | - |
| **VA-GS (Ours)** | **0.49** | 15.5m |

### TNT Dataset Reconstruction (F1-score ↑)

| Method | Barn | Truck | Mean |
|--------|------|-------|------|
| 3DGS | 0.13 | 0.19 | 0.09 |
| PGSR | 0.66 | 0.66 | 0.52 |
| GausSurf | 0.50 | 0.65 | 0.50 |
| **VA-GS (Ours)** | **0.71** | 0.64 | **0.54** |

### Ablation Study (TNT F1-score)

| Configuration | Precision | Recall | F1 | Note |
|--------------|-----------|--------|----|------|
| $\mathcal{L}_I$ only | 0.09 | 0.23 | 0.13 | No geometric constraint, worst |
| w/o $\mathcal{L}_{nc}+\mathcal{L}_{ns}$ | 0.40 | 0.57 | 0.46 | Normal constraints are critical |
| w/o $\mathcal{L}_p+\mathcal{L}_f$ | 0.33 | 0.40 | 0.36 | Multi-view alignment is indispensable |
| + scale loss (planarization) | 0.51 | 0.60 | 0.54 | No additional gain |
| Full model | 0.51 | 0.60 | **0.54** | Modules are complementary |

### Key Findings

- Flattening 3D Gaussians into planar discs (scale loss) yields no benefit for this method, and even degrades rendering quality on Mip-NeRF 360 — indicating that preserving full 3D Gaussian representation is preferable.
- Using $N=3$ source views for multi-view alignment is the optimal trade-off; $N=4$ provides no additional gain but increases computational cost.
- Removing either photometric or feature alignment individually has limited impact, but removing both causes F1 to drop sharply from 0.54 to 0.36.

## Highlights & Insights

1. **No external geometric representation** (e.g., SDF, mesh) is required; high-quality reconstruction is achieved purely by constraining the Gaussian primitives themselves.
2. The visibility and occlusion weighting design is highly practical yet effective; using reprojection error as an occlusion criterion is both concise and elegant.
3. The strategy of down-weighting normal loss at edges ($\delta=(1-\nabla I)^2$) directly and effectively resolves boundary normal ambiguity.

## Limitations & Future Work

- Training is relatively slow (15.5m vs. 3.4m for 3DGS) due to the computational overhead of multi-view alignment.
- The normal smoothness constraint may over-smooth regions of high curvature.
- The depth image feature alignment depends on the quality of the pretrained model.

## Related Work & Insights

- Multi-view photometric consistency is a classical approach for addressing the insufficiency of single-view supervision; the key challenge lies in handling occlusion and visibility.
- The robustness of feature-level alignment to illumination variation is worth borrowing in other tasks such as SLAM and SfM.

## Rating

- Novelty: ⭐⭐⭐⭐ Each module is well-designed; classical MVS ideas are organically integrated with 3DGS.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of DTU/TNT/Mip-NeRF360 with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Formulations are clearly derived, though notation is occasionally dense.
- Value: ⭐⭐⭐⭐ Practically valuable for surface reconstruction with 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene](haif-gs_hierarchical_and_induced_flow-guided_gaussian_splatting_for_dynamic_scen.md)
- [\[NeurIPS 2025\] SceneForge: Enhancing 3D-text alignment with Structured Scene Compositions](sceneforge_enhancing_3d-text_alignment_with_structured_scene_compositions.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](../../CVPR2026/3d_vision/dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
