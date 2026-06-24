---
title: >-
  [Paper Note] Feature-Preserving Mesh Decimation for Normal Integration
description: >-
  [CVPR 2025][3D Vision][mesh decimation] The classic quadric error metric (QEM) is derived in screen space with normal maps as input. Combined with optimal Delaunay triangulation, this achieves anisotropic mesh decimation that preserves sub-millimeter accuracy even at 90%+ compression rates, accelerating high-resolution normal integration from hours to minutes.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "mesh decimation"
  - "normal integration"
  - "quadric error metric"
  - "anisotropic meshing"
  - "Delaunay triangulation"
date: 2026-05-08
content_hash: 1c0177798cdd9d24
---

# Feature-Preserving Mesh Decimation for Normal Integration

**Conference**: CVPR 2025  
**arXiv**: [2504.00867](https://arxiv.org/abs/2504.00867)  
**Code**: [https://moritzheep.github.io/anisotropic-screen-meshing](https://moritzheep.github.io/anisotropic-screen-meshing)  
**Area**: 3D Vision  
**Keywords**: mesh decimation, normal integration, quadric error metric, anisotropic meshing, Delaunay triangulation

## TL;DR

The classic quadric error metric (QEM) is derived in screen space with normal maps as input. Combined with optimal Delaunay triangulation, this achieves anisotropic mesh decimation that preserves sub-millimeter accuracy even at 90%+ compression rates, accelerating high-resolution normal integration from hours to minutes.

## Background & Motivation

**Background**: Normal integration reconstructs 3D surfaces from estimated normal maps (e.g., acquired via photometric stereo), typically by solving the Poisson equation on pixel-level dense meshes. While normal maps capture pixel-level surface details, the computational cost at high resolutions scales quadratically with the number of pixels.

**Limitations of Prior Work**:
- Pixel-level dense meshes at high resolutions (e.g., 16MP) lead to normal integration runtimes of several hours.
- Doubling the resolution requires doubling both the width and height of the image, causing a quadratic increase in the number of variables.
- Existing adaptive triangular meshing methods [Heep et al.] only support isotropic decimation (equilateral triangles), failing to align accurately with ridges and valleys.
- Fundamental problem: In regular grids, detailed regions require high resolution, which forces flat regions to maintain the same high resolution.

**Key Challenge**: In pixel grids, local detail requirements force global resolution increases. While triangular meshes allow local adaptivity, isotropic decimation lacks accuracy at geometric features.

**Goal**: The goal is to design an anisotropic mesh decimation algorithm that aligns the vertices and edges of a triangular mesh along surface ridges and valleys, preserving geometric accuracy even at extremely high compression rates.

**Key Insight**: Derive the QEM from 3D mesh decimation in 2D screen space (using normal maps instead of 3D geometry) and leverage generalized Delaunay triangulation to achieve anisotropic edge alignment.

**Core Idea**: Prior to normal integration, a normal-map-driven screen-space QEM and generalized Delaunay triangulation are utilized for anisotropic mesh preprocessing, faithfully preserving geometric features with minimal vertices.

## Method

### Overall Architecture

1. Initialize the foreground pixel grid as a dense triangular mesh (two triangles per pixel).
2. Iteratively execute three steps: edge collapse (removing redundant vertices) $\to$ edge flip (aligning ridges) $\to$ vertex relocation (precise positioning).
3. Perform normal integration on the decimated sparse triangular mesh (solving a linear system to obtain the depth map).

Objective function: $E = E_{Geo} + \lambda \cdot E_{ODT}$, where $E_{Geo}$ estimates the deviation of the mesh from the underlying surface (QEM), and $E_{ODT}$ ensures a uniform distribution in flat regions (Delaunay regularization).

### Key Designs

**1. Screen-Space Quadric Error Metric**
- **Function**: Derive QEM directly from the normal map instead of relying on known 3D geometry. Surface tangents (Jacobian $J_f$) are calculated from normals to convert 3D surface integration into screen-coordinate integration.
- **Mechanism**: For each vertex $v$, compute its quadric $Q_v(\delta\vec{x}) = \sum_{f \in \mathcal{F}_v} \frac{A_f^{(3)}}{|\mathcal{P}_f|} \sum_{p \in \mathcal{P}_f} \|J_f \delta\vec{u}_{vp} + \delta\vec{x}\|_{M_p}^2$, where $M_p = \vec{n}_p \cdot \vec{n}_p^t + \lambda \cdot \mathbb{1}$ unifies geometric error and Delaunay regularization.
- **Design Motivation**: The runtime advantage depends on decimating the mesh before integration; thus, decimation cannot wait for the 3D geometry to be known. The chain of relationships from normal $\to$ tangent $\to$ Jacobian allows the entire computation to be performed in screen space.

**2. Generalized Delaunay Edge Flipping (Edge Alignment)**
- **Function**: Extend standard Delaunay triangulation to anisotropic scenarios using the $\|\cdot\|_M$ norm, deciding whether to flip edges based on a local tetrahedral criterion.
- **Mechanism**: For each non-boundary edge $e$, the four adjacent vertices are projected onto the tangent plane to obtain a flat approximation, which is then "lifted" into a tetrahedron using $\|\cdot\|_{M_e}$. If the current edge is not the shortest edge of the tetrahedron (i.e., the other diagonal is more "geometrically aligned"), the edge is flipped.
- **Design Motivation**: Standard Delaunay is isotropic and cannot align edges along ridges and valleys. The outer product of normals $\vec{n}\cdot\vec{n}^t$ in the generalized norm $M_e$ introduces anisotropy—stretching in directions where normal gradients are high (at features) to naturally align edges with geometric features.

**3. QEM-driven Edge Collapse (Decimation)**
- **Function**: Quantify the impact of each collapse on surface shape using the quadric cost $C_{vw} = \min_{\vec{u}_{vw}} \tilde{Q}_v(\vec{u} - \vec{u}_v) + \tilde{Q}_w(\vec{u} - \vec{u}_w)$, prioritizing collapsing the edge with the lowest cost.
- **Mechanism**: All collapsible edges are placed in a priority queue sorted by ascending cost. After a collapse, the sum of the quadrics of the two endpoints serves as the quadric for the new vertex.
- **Design Motivation**: Follows the classic design of Garland & Heckbert, but using the screen-space quadric version derived from normal maps.

### Loss & Training

- Non-learning method, no training required.
- Hyperparameters: $\lambda = 10^{-5}$ (trade-off between geometric error and Delaunay regularization), $\alpha = 0.5$ (step size for vertex relocation).
- 5 iterations: each round performs edge collapse followed by edge flipping and vertex alignment.
- Two resolution control modes are supported: target vertex count or collapse cost threshold.
- C++ implementation, using nvdiffrast for triangle-to-pixel mapping, and Eigen to solve the linear system.

## Key Experimental Results

### Main Results (DiLiGenT-MV dataset, 20-view average RMSE, mm)

| Object | Pixel-based [2] | Isotropic low | Isotropic high | **Ours low** | **Ours high** |
|---|---|---|---|---|---|
| Bear | 2.97 | 3.95 | 3.37 | **3.84** | **3.04** |
| Buddha | 6.74 | 7.74 | 7.33 | **6.86** | **6.61** |
| Cow | 2.45 | 3.42 | 2.96 | **3.07** | **2.74** |
| Pot2 | 5.15 | 5.89 | 5.65 | **5.63** | **5.29** |
| Reading | 6.34 | 7.08 | 6.83 | **6.82** | **6.50** |

### Ablation Study

| Configuration | Effect |
|---|---|
| Edge collapse only | Retains coarse shape, high-frequency details are lost |
| Collapse + vertex alignment | RMSE improves, vertices are localized at features |
| Collapse + edge alignment | MAE improves, edges are aligned along ridges/valleys |
| **Full algorithm** | Both RMSE and MAE are optimal |

### Key Findings

1. **Sub-millimeter accuracy at 90%+ compression**: Even with over 95% compression (where vertex count is only 5% of the pixel count), fine surface details remain clearly visible.
2. **Anisotropic outperforms isotropic**: At the same vertex count, the proposed method consistently outperforms isotropic methods in both RMSE and MAE, visually preserving more details (e.g., the bear's nose, the Buddha's face).
3. **"Less is more"**: Isotropic methods accumulate more vertices at ridges but yield lower accuracy due to the lack of edge alignment. The proposed method achieves higher accuracy with fewer vertices but a superior layout.
4. **Handles up to 16MP normal maps**: Within the same timeframe, while pixel-based methods can only process 4MP, the proposed method can triangulate, decimate, and integrate 16MP maps.

## Highlights & Insights

- The theoretical contribution of deriving the classic QEM from 3D geometry to screen-space normal maps is clean and elegant.
- Discovers the intrinsic connection between QEM and generalized Delaunay triangulation under a unified norm $\|\cdot\|_M$, unifying geometry preservation and edge alignment into a single framework.
- Simple algorithmic design with only three local operations (collapse, flip, relocation).
- Counterintuitive insight: Using fewer but more precisely positioned vertices at features outperforms having dense but isotropic vertices.

## Limitations & Future Work

- The weak perspective projection assumption (assuming a constant camera-object distance) may introduce errors in scenes with large depth variations.
- The quality of the normal map directly determines the decimation quality; robustness to noisy normals is not fully addressed.
- The C++ implementation is reasonably efficient but does not leverage GPU parallelism.
- Only targeted at normal integration scenarios and not yet extended to other mesh decimation applications.
- Low-resolution datasets [DiLiGenT-MV, LUCES] or the lack of ground-truth geometry [PS, RGBN] limit the comprehensiveness of the evaluation.

## Related Work & Insights

- Garland & Heckbert's QEM is a classic method for mesh decimation; this work is the first to derive it in screen space without 3D geometry.
- Anisotropic extensions of CVT/ODT typically require lifting vertices and normals to a 6D space; the proposed method operates more directly in 2D screen space.
- Insight: Normals, as a rich geometric prior, can drive the "preprocessing" of more traditional geometry processing algorithms.

## Rating

⭐⭐⭐⭐ — Elegant theoretical derivation, significant experimental results, but the application scenario is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Variational Graph-based Normal Integration](../../CVPR2026/3d_vision/variational_graph-based_normal_integration.md)
- [\[CVPR 2025\] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)
- [\[CVPR 2025\] 3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)
- [\[CVPR 2025\] Geometry in Style: 3D Stylization via Surface Normal Deformation](geometry_in_style_3d_stylization_via_surface_normal_deformation.md)
- [\[CVPR 2025\] Identity-preserving Distillation Sampling by Fixed-Point Iterator](identity-preserving_distillation_sampling_by_fixed-point_iterator.md)

</div>

<!-- RELATED:END -->
