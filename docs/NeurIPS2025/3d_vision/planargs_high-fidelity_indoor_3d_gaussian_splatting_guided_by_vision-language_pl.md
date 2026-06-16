---
title: >-
  [Paper Note] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] PlanarGS detects planar regions via a vision-language foundation model (GroundedSAM) with text prompts, combines multi-view depth priors from DUSt3R…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Indoor Reconstruction"
  - "Planar Priors"
  - "Vision-Language Models"
  - "DUSt3R"
date: 2026-05-08
content_hash: bad283abe9b5cdea
---

# PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors

**Conference**: NeurIPS 2025
**arXiv**: [2510.23930](https://arxiv.org/abs/2510.23930)  
**Code**: [Project Page](https://planargs.github.io)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Indoor Reconstruction, Planar Priors, Vision-Language Models, DUSt3R

## TL;DR

PlanarGS detects planar regions via a vision-language foundation model (GroundedSAM) with text prompts, combines multi-view depth priors from DUSt3R, and optimizes 3DGS through coplanarity constraints and geometric prior supervision to achieve high-fidelity surface reconstruction in indoor scenes.

## Background & Motivation

3D reconstruction of indoor scenes is widely demanded in AR/VR and robotics, yet the central challenge lies in **large low-texture regions** (e.g., walls, floors, ceilings). 3DGS relies on photometric loss for training, which introduces severe geometric ambiguity in such regions:
- Existing methods (e.g., PGSR) apply multi-view geometric consistency constraints but achieve only local smoothness;
- Incorporating monocular depth/normal priors (e.g., DN-Splatter) suffers from local misalignment;
- All such methods impose **local constraints only**, failing to guarantee global planar flatness — resulting in the typical "locally smooth but globally curved" artifact.

The ideal solution is to **explicitly detect planar regions and enforce planar geometry**. However, traditional plane detectors (e.g., PlaneRCNN and similar specialized small models) exhibit poor generalization and low segmentation accuracy.

The core ideas of this paper are:
1. Leveraging **vision-language foundation models** (GroundedSAM) with text prompts (e.g., "wall," "floor") to detect planar regions — exploiting the generalization of foundation models and the flexibility of text prompting;
2. Correcting detection errors through **cross-view fusion and geometric verification**;
3. Applying coplanarity constraints to **globally** regularize the Gaussian distribution over planar regions.

## Method

### Overall Architecture

Multi-view images → DUSt3R extracts multi-view depth/normal priors → GroundedSAM + LP3 pipeline generates planar priors → 3DGS optimization (planar prior supervision + geometric prior supervision) → TSDF fusion for mesh extraction.

### Key Designs

1. **LP3: Language-Prompted Planar Prior Pipeline**

    - **Cross-view fusion**: Large planes in a single image may extend beyond the field of view and be missed. LP3 back-projects planar masks from neighboring frames into the current frame using prior depth, compensating for missed detections.
    - **Geometric verification**: GroundedSAM occasionally merges two perpendicular walls into a single mask. LP3 first computes a normal map $N_{dr}$ from the depth prior, then: (1) applies K-means clustering on the normal map to separate non-parallel planes; (2) uses outlier detection on the plane distance map $\delta_r = P \cdot N_{dr}$ to identify geometric boundaries and separate parallel but distinct planes.
    - **Design motivation**: Raw VL model segmentation suffers from two failure modes — missed detections (due to field-of-view limits) and over-merging (two planes treated as one). LP3 corrects these via multi-view complementation and geometric verification.

2. **Planar Prior Supervision**

    - **Plane-guided initialization**: SfM produces sparse point clouds in low-texture regions; prior depth is used to back-project planar pixels into dense 3D points to supplement initialization.
    - **Gaussian flattening**: The minimum scaling factor of each Gaussian is minimized via $L_s = \|min(s_1,s_2,s_3)\|_1$, encouraging Gaussians to collapse into flat discs.
    - **Coplanarity constraint** (core): The rendered depth map is back-projected into 3D points; for each planar region $p_m$, plane parameters $A_m$ are fitted by least squares ($A_m^T P = 1$); planar depth is then recovered as $D_p(p) = (A_m^T K^{-1} \tilde{p})^{-1}$; the rendered depth $\hat{D}$ is constrained to be consistent with the planar depth $D_p$: $L_p = \frac{1}{N_p}\sum \|D_p - \hat{D}\|_1$.

3. **Geometric Prior Supervision**

    - **Prior depth constraint**: DUSt3R depth is scale-shift aligned with SfM sparse depth and used to supervise rendered depth in low-texture regions.
    - **Prior normal constraint**: Rendered surface normals in planar regions are constrained to align with DUSt3R prior normals.
    - **Depth-normal consistency**: In low-texture regions, rendered GS-normals are constrained to be consistent with surface normals derived from rendered depth.

### Loss & Training

$$L_{total} = L_{RGB} + L_s + \lambda_1 L_{dn} + \lambda_2 L_p + \lambda_3 L_{rd} + \lambda_4 L_{rn}$$

$\lambda_1=0.05, \lambda_2=0.5, \lambda_3=0.05, \lambda_4=0.2$. Training runs for 30K iterations and completes within one hour on an RTX 3090.

## Key Experimental Results

### Main Results: MuSHRoom Dataset (5 complex real-world scenes)

| Method | Acc↓ | Comp↓ | CD↓ | F1↑ | NC↑ | PSNR↑ |
|--------|------|-------|-----|-----|-----|-------|
| 3DGS | 12.01 | 11.85 | 11.92 | 38.53 | 62.00 | 25.79 |
| DN-Splatter | 6.25 | 5.29 | 5.77 | 61.86 | 77.13 | 24.80 |
| **PlanarGS** | **3.95** | **5.02** | **4.49** | **77.14** | **83.35** | **26.42** |

### ScanNet++ and Replica Datasets

| Method | ScanNet++ CD↓ | ScanNet++ F1↑ | Replica CD↓ | Replica F1↑ |
|--------|--------------|--------------|-------------|-------------|
| DUSt3R | 8.17 | 38.17 | 7.35 | 44.89 |
| PGSR | 7.22 | 53.73 | 8.56 | 62.98 |
| DN-Splatter | 4.16 | 75.86 | 5.60 | 68.12 |
| **PlanarGS** | **3.66** | **82.78** | **4.13** | **81.90** |

### Ablation Study (MuSHRoom coffee room)

| Configuration | Acc↓ | F1↑ | Notes |
|---------------|------|-----|-------|
| ZeroPlane as planar prior | — | Lower | Specialized small model generalizes poorly; erroneous priors introduce noise |
| GroundedSAM (w/o LP3) | — | Medium | Priors without geometric verification are insufficiently accurate |
| w/o coplanarity constraint | — | Reduced | Surface undulation appears on large planar regions |
| w/o geometric prior | — | Reduced | Lack of scale supervision causes global planar tilt |
| w/o depth-normal consistency | — | Reduced | Surface roughness increases |
| Full PlanarGS | Best | Best | All modules are complementary |

### Key Findings

- The coplanarity constraint contributes the most individually (especially in the absence of geometric priors), but omitting geometric priors leads to **global tilt and offset** of large planes.
- The cross-view fusion and geometric verification in the LP3 pipeline are critical for planar prior quality — ZeroPlane and bare GroundedSAM are both insufficient.
- Training time is comparable to other 3DGS methods (<1 hour), confirming practical applicability.

## Highlights & Insights

1. A representative example of **empowering classical tasks with foundation models** — leveraging the generalization of VL models for prior extraction in geometric reconstruction.
2. Flexibility of text prompts: adding "blackboard" for a classroom detects blackboards without any retraining.
3. The "global" formulation of the coplanarity constraint is noteworthy — rather than imposing per-pixel normal constraints, it fits a plane globally and then constrains depth consistency.

## Limitations & Future Work

- Preprocessing relies on two large models (DUSt3R and GroundedSAM), increasing deployment complexity.
- Text prompts require manual specification, despite their relatively broad applicability.
- Non-planar regions still depend on conventional photometric loss; improvements for complex non-planar objects (e.g., plants, fabrics) remain limited.

## Related Work & Insights

- The LP3 pipeline (VL detection + cross-view fusion + geometric verification) is transferable to other 3D tasks requiring semantic priors.
- The approach of fitting planes via least squares and back-constraining depth is concise and effective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to introduce VL foundation model planar priors into 3DGS; LP3 pipeline is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three indoor datasets plus ablations, covering both synthetic and real scenes.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with thorough method descriptions.
- Value: ⭐⭐⭐⭐⭐ A practical method for indoor reconstruction that substantially outperforms prior SOTA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](../../ICCV2025/3d_vision/autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[NeurIPS 2025\] LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS](langsplatv2_high-dimensional_3d_language_gaussian_splatting_with_450_fps.md)
- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](../../ICCV2025/3d_vision/gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Plana3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting](plana3r_zero-shot_metric_planar_3d_reconstruction_via_feed-forward_planar_splatt.md)

</div>

<!-- RELATED:END -->
