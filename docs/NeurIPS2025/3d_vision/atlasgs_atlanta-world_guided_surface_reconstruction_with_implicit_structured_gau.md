---
title: >-
  [Paper Note] AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] AtlasGS is proposed to achieve smooth, high-frequency-detail-preserving surface reconstruction in indoor and urban scenes by incorporating the Atlanta-world structural pri…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "surface reconstruction"
  - "Atlanta-world assumption"
  - "implicit representation"
  - "indoor/urban scenes"
date: 2026-05-08
content_hash: 1fef22cb804affc6
---

# AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians

**Conference**: NeurIPS 2025
**arXiv**: [2510.25129](https://arxiv.org/abs/2510.25129)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, surface reconstruction, Atlanta-world assumption, implicit representation, indoor/urban scenes

## TL;DR

AtlasGS is proposed to achieve smooth, high-frequency-detail-preserving surface reconstruction in indoor and urban scenes by incorporating the Atlanta-world structural prior into an implicit-structured Gaussian representation, comprehensively outperforming existing implicit and explicit methods.

## Background & Motivation

**Indoor/urban reconstruction is a hot topic**: Applications such as digital twins, robot navigation, and augmented reality demand high-precision and efficient reconstruction.

**Low-texture regions are the core challenge**: Floors, ceilings, and plain walls in man-made scenes lack texture features; traditional multi-view stereo methods fail in these regions, producing incomplete or distorted geometry.

**Monocular geometric priors lack global consistency**: Monocular depth/normal priors provide only local smoothness signals and are frequently inconsistent across viewpoints, resulting in uneven surfaces.

**The Manhattan-world assumption is overly restrictive**: It requires scenes to be aligned along three orthogonal directions, failing to handle non-orthogonally arranged buildings in urban scenes (e.g., oblique structures).

**The discreteness of 2DGS leads to surface discontinuities**: 2D Gaussian Splatting optimizes surfel primitives independently, producing broken surfaces in low-texture or under-observed regions.

**Naive combination of implicit SDF and GS performs poorly**: Existing methods (e.g., GSRec) attempt to regularize Gaussian optimization with an implicit SDF field, but mutual interference between the two typically degrades reconstruction quality.

**Core motivation**: There is a need for (1) globally consistent geometric priors to regularize low-texture regions, and (2) a 3D representation that combines the efficiency and high-frequency detail preservation of Gaussians with the smoothness of implicit methods.

## Method

### Overall Architecture

Given posed multi-view images and an SfM point cloud, a sparse feature voxel grid is constructed and the scene is represented as implicit-structured 2D Gaussians (surfels). Gaussian attributes are predicted via an attribute decoder and a semantic decoder; after rasterization, supervision is applied through RGB images, monocular geometric priors, and semantic labels. Learnable plane indicators based on the Atlanta-world assumption are simultaneously introduced to constrain global structure.

### Three Core Designs

#### 1. Implicit-Structured Gaussian Representation

- A **sparse feature voxel grid** $\mathcal{V}$ is built from the SfM point cloud. Each voxel contains geometric features $\mathcal{V}_g$, semantic features $\mathcal{V}_s$, offsets $\Delta_k$ for $\mathcal{K}=10$ local Gaussians, and a shared scaling factor $l$.
- A geometric MLP $\mathcal{M}_g$ decodes opacity $\alpha$, scale $s$, rotation $q$, and (view-dependent) color $c$; a semantic MLP $\mathcal{M}_s$ decodes semantic attributes $z \in \mathbb{R}^4$ (wall/floor/ceiling/other).
- Gaussian positions are computed as $\mathbf{p}_k^i = \mathbf{v}_i + l \cdot \Delta_k^i$, i.e., voxel center plus offset.
- **Core advantage**: The shared decoder causes each Gaussian's optimization to implicitly influence its neighborhood, achieving local geometric consistency while preserving high-frequency details via Gaussian primitives — in contrast to the independent per-primitive optimization of 2DGS.

#### 2. Gaussian Semantic Lifting

- A pretrained semantic segmentation model generates 2D pseudo-labels $\hat{Z}$ across four categories: wall, floor, ceiling, and other.
- The 3D semantic attributes $z$ are rendered into image space to obtain semantic probabilities $Z$, optimized via a cross-entropy loss $\mathcal{L}_{\text{sem}}$.
- A stop-gradient operation blocks the backpropagation of semantic supervision into geometry optimization, preventing inconsistent labels from corrupting geometric optimization.

#### 3. Atlanta-world Guided Planar Regularization

**Learnable plane indicators**: A floor plane $\pi_f = (\mathbf{n}_g, d_f)$ and a ceiling plane $\pi_c = (-\mathbf{n}_g, d_c)$ are defined, where $\mathbf{n}_g$ is the gravity direction and $d_f, d_c$ are distances from the origin. Ceiling planes are omitted for outdoor urban scenes. The planes are initialized via RANSAC and jointly optimized with the Gaussians.

**3D global planar regularization** $\mathcal{L}_{3D}$:

- **Normal alignment**: Wall Gaussian normals should be perpendicular to the gravity direction ($1 - |\mathbf{n}_g^\top \mathbf{n}_i|$); floor/ceiling Gaussian normals should be parallel to it ($|\mathbf{n}_g^\top \mathbf{n}_i|$).
- **Planar constraint**: Floor/ceiling Gaussian positions should lie on the corresponding plane ($|d_f + \mathbf{n}_g^\top \mathbf{p}_i|$, etc.).
- All terms are weighted by semantic probabilities as soft constraints.

**2D local surface regularization** $\mathcal{L}_{2D}$:

- For wall regions: the explicit decoupling of Gaussian positions and normals means that optimizing normals alone cannot constrain spatial distribution.
- 3D points are back-projected from the rendered depth map, and local surface normals $\mathbf{N}_d$ are computed to constrain their relationship with the gravity direction.
- Terms are similarly weighted by semantic probabilities to mitigate the effect of semantic misclassification.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_1 \mathcal{L}_{\text{depth}} + \lambda_2 \mathcal{L}_{\text{normal}} + \lambda_3 \mathcal{L}_{\text{reg}} + \lambda_4 \mathcal{L}_{\text{sem}} + \lambda_5 \mathcal{L}_{\text{dist}} + \lambda_6 \mathcal{L}_{\text{nc}}$$

where $\mathcal{L}_{\text{reg}} = \mathcal{L}_{3D} + \mathcal{L}_{2D}$; $\mathcal{L}_{\text{depth}}$ aligns monocular depth priors via scale-shift L2 loss; $\mathcal{L}_{\text{normal}}$ jointly constrains both the rendered normals and the depth-derived normals to be consistent with the prior normals.

## Key Experimental Results

### Datasets & Baselines

- **Indoor**: Replica (7 synthetic scenes), ScanNet (4 real scenes), ScanNet++ (4 real scenes)
- **Outdoor**: MatrixCity (4 city blocks, synthetic)
- **Baselines**: Implicit methods (ManhattanSDF, MonoSDF); explicit methods (Scaffold-GS, 2DGS, DN-Splatter, GSRec); GaussianPro added for outdoor

### Main Results

| Dataset | Metric | AtlasGS | Best Baseline | Gain |
|--------|------|---------|----------|------|
| Replica | F-score ↑ | **87.35** | MonoSDF 73.08 | +14.27 |
| ScanNet++ | F-score ↑ | **87.48** | ManhattanSDF 76.67 | +10.81 |
| ScanNet | F-score ↑ | **77.98** | MonoSDF 71.21 | +6.77 |
| ScanNet | Acc ↓ (cm) | **3.62** | ManhattanSDF 4.25 | −0.63 |
| MatrixCity | CD ↓ | **0.028** | GaussianPro 0.091 | −0.063 |

### Key Findings

1. **Comprehensively outperforms both implicit and explicit methods**: F-score leads by large margins on all indoor datasets, with both accuracy and completeness surpassing baselines.
2. **More efficient than implicit methods**: Training takes 27 minutes on ScanNet vs. 7+ hours for implicit methods; rendering runs at 70 FPS vs. <10 FPS.
3. **Effective on outdoor scenes**: CD of only 0.028 on MatrixCity, far superior to all baselines (including GSRec at 0.112 and 2DGS at 0.106).
4. **Competitive novel-view synthesis quality**: Though not best overall (PSNR 39.58 vs. 2DGS 41.59 on Replica), AtlasGS achieves the best LPIPS (0.2517) on the real-world ScanNet++ dataset, with geometric accuracy producing fewer artifacts.

### Ablation Study (ScanNet)

| Configuration | CD ↓ | F-score ↑ |
|------|------|-----------|
| 2DGS + depth/normal priors | 12.68 | 39.27 |
| Implicit-structured GS (w/o $\mathcal{L}_{\text{reg}}$) | 4.10 | 74.23 |
| + $\mathcal{L}_{3D}$ (w/o $\mathcal{L}_{2D}$) | 3.97 | 75.52 |
| Full model | **3.77** | **77.98** |

- The implicit-structured representation alone substantially improves quality (F-score: 39.27 → 74.23).
- The 3D and 2D regularization terms each contribute approximately 1–2 F-score points, for a combined gain of 3.75.
- Removing either the depth or normal prior degrades performance, confirming the indispensability of geometric priors.

## Highlights & Insights

1. **The Atlanta-world assumption is more general than Manhattan-world**: It permits multiple non-orthogonal horizontal directions, unifying indoor and urban scenes under a single structural prior — a well-motivated and practically useful extension.
2. **The implicit-structured Gaussian design is elegant**: Rather than naively stacking implicit and explicit representations, the voxel grid is embedded within the Gaussian framework, achieving local consistency through a shared MLP decoder and avoiding the mutual interference seen in prior methods.
3. **Semantic–geometry decoupling**: The stop-gradient blocking of semantic supervision from geometry backpropagation is a subtle yet important design choice.
4. **Insight behind 2D local surface regularization**: The paper identifies that the decoupling of normals and positions in Gaussian representations means that constraining normals alone is insufficient; local surface normals must instead be derived from the rendered depth map to indirectly constrain positions.

## Limitations & Future Work

1. **Slower training and rendering than pure Gaussian methods**: 27 minutes vs. 11–12 minutes for training; 70 FPS vs. 118–279 FPS for rendering. Decoding all Gaussian attributes via MLP introduces significant overhead.
2. **Dependency on pretrained semantic segmentation models**: Semantic categories are fixed to four classes (wall/floor/ceiling/other), limiting applicability to atypical structural scenes (e.g., curved buildings, natural environments).
3. **Inherent scope of the Atlanta-world assumption**: It applies only to man-made scenes with a dominant gravity direction and planar structures, and is unsuitable for natural terrain or unstructured environments.
4. **Novel-view synthesis is not state-of-the-art**: PSNR on the synthetic Replica dataset is lower than 2DGS, indicating that geometric constraints impose a certain cost on rendering quality.

## Related Work & Insights

- **Implicit surface reconstruction**: NeRF → NeuS/VolSDF (SDF + volume rendering) → incorporation of monocular priors (MonoSDF) and semantics (ManhattanSDF). Limited by MLP capacity and training speed.
- **Gaussian surface reconstruction**: 3DGS → 2DGS/Gaussian Surfels (surfel primitives for improved multi-view consistency) → PGSR (planar Gaussians) → GSRec (IMLS regularization) → DN-Splatter (depth-normal priors). Discreteness remains the core issue.
- **Structural priors**: Manhattan-world (three orthogonal directions) → Atlanta-world (one gravity direction + multiple horizontal directions); the latter is more flexible.
- **Joint implicit–explicit methods**: NeuSG, GSDF, etc. learn SDF and GS simultaneously, but mutual interference degrades results. The embedded design in this paper avoids this problem.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing the Atlanta-world assumption into Gaussian splatting is a novel combination, and the implicit-structured Gaussian design is original; however, the individual technical components (voxel grids, MLP decoding, semantic lifting) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets (indoor and outdoor), 6+ baselines, and comprehensive ablations yield convincing results. Failure case analysis on complex, unstructured scenes is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, methodology is described in detail, and figures are high quality. Equations are numerous but well organized.
- **Value**: ⭐⭐⭐⭐ — Achieves comprehensive state-of-the-art performance on the important indoor/urban reconstruction task, with high engineering and academic value. Primarily limited by speed and applicable scene scope.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction](geosvr_taming_sparse_voxels_for_geometrically_accurate_surface_reconstruction.md)
- [\[ICCV 2025\] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](../../ICCV2025/3d_vision/rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)
- [\[CVPR 2026\] NTK-Guided Implicit Neural Teaching](../../CVPR2026/3d_vision/ntk-guided_implicit_neural_teaching.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[NeurIPS 2025\] Linearly Constrained Diffusion Implicit Models](linearly_constrained_diffusion_implicit_models.md)

</div>

<!-- RELATED:END -->
