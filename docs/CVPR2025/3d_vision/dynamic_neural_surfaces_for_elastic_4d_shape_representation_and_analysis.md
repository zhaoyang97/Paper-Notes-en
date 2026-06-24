---
title: >-
  [Paper Note] Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis
description: >-
  [CVPR 2025][3D Vision][4D shape analysis] This paper proposes Dynamic Spherical Neural Surfaces (D-SNS), which model genus-0 4D surfaces as spatio-temporal continuous functions using MLPs. Spatio-temporal registration, geodesic calculation, and mean estimation are directly performed in SRNF/SRVF spaces without discretization, outperforming 4D Atlas on 4D human and facial datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D shape analysis"
  - "neural surface representation"
  - "spatio-temporal registration"
  - "Riemannian geometry"
  - "elastic metric"
date: 2026-05-08
content_hash: f0b99dead02e65cd
---

# Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis

**Conference**: CVPR 2025  
**arXiv**: [2503.03132](https://arxiv.org/abs/2503.03132)  
**Code**: [https://4d-dsns.github.io/DSNS/](https://4d-dsns.github.io/DSNS/)  
**Area**: Human Understanding / 3D Vision  
**Keywords**: 4D shape analysis, neural surface representation, spatio-temporal registration, Riemannian geometry, elastic metric

## TL;DR
This paper proposes Dynamic Spherical Neural Surfaces (D-SNS), which model genus-0 4D surfaces as spatio-temporal continuous functions using MLPs. Spatio-temporal registration, geodesic calculation, and mean estimation are directly performed in SRNF/SRVF spaces without discretization, outperforming 4D Atlas on 4D human and facial datasets.

## Background & Motivation

**Background**: 4D shape analysis (the statistical analysis of 3D surfaces deforming over time) is an important topic in computer vision and computer graphics. Traditional methods discretize 4D surfaces into sequences of triangular meshes, computing registration, geodesics, and statistics on discrete points. While parametric models like SMPL can handle human motion, they are category-specific and cannot perform cross-category analysis.

**Limitations of Prior Work**: The performance of traditional discretization methods depends heavily on the quality and resolution of mesh sampling. When temporal sampling becomes sparse, the accuracy of discrete methods drops significantly. Furthermore, a pipeline that discretizes first and analyzes later can lead to sub-optimal solutions. Nonlinear optimization under elastic metrics is computationally expensive in the discrete domain.

**Key Challenge**: 4D shape analysis requires simultaneously solving spatial registration (different parametrizations) and temporal registration (different deformation speeds). Existing methods must discretize first to perform these operations, resulting in information loss between the two steps.

**Goal**: To represent 4D surfaces as spatially and temporally continuous functions, allowing spatio-temporal registration, geodesics, and statistics calculation to be performed directly in the continuous domain.

**Key Insight**: Drawing inspiration from neural implicit representations (such as NeRF and NeuS), the authors fit genus-0 surfaces with spherical parameterization as continuous mappings using MLPs. The key observation is that, for surfaces with spherical parameterization, the SRNF mapping can simplify complex elastic metrics into the simple $\mathbb{L}^2$ metric.

**Core Idea**: Use MLPs to encode 4D surfaces as continuous functions mapping $\mathbb{S}^2 \times [0,1] \to [-1,1]^3$ (D-SNS), and then perform all shape analysis tasks directly in the continuous domain using SRNF/SRVF spaces.

## Method

### Overall Architecture
The input consists of discrete 4D surfaces (genus-0 triangular mesh sequences), which are fitted into a continuous D-SNS representation using an MLP after spherical parameterization. Spatial registration is then conducted in the SRNF space, and temporal registration is performed in the SRVF space, leading to geodesic and Karcher mean computation. All analysis tasks are completed directly on the neural representation, and discretization is only used for visualization.

### Key Designs

1. **Dynamic Spherical Neural Surfaces (D-SNS)**:

    - **Function**: Encodes discrete 4D surfaces into spatio-temporally continuous functions.
    - **Mechanism**: An MLP $F_\Theta: \mathbb{S}^2 \times [0,1] \to [-1,1]^3$ maps spherical coordinates $s$ and time $t$ to 3D surface points. The network consists of 6 residual blocks, each containing two layers of 1024 nodes, using the SoftPlus activation function to ensure smoothness. Positional encoding is applied to both spatial and temporal domains. During training, the MSE between predicted and ground-truth points is minimized with a batch size of 80K points.
    - **Design Motivation**: Continuous representations allow differential quantities such as normal and tangent fields to be computed directly via automatic differentiation, eliminating the need for discrete approximations. Residual connections significantly improve the representation capability for both coarse and fine geometric details.

2. **Spatial Registration (SRNF Space)**:

    - **Function**: Aligns different parametrizations between two 3D surfaces.
    - **Mechanism**: Maps surfaces to the Square Root Normal Fields (SRNF) space, where the elastic metric becomes an $\mathbb{L}^2$ metric. The spatial diffeomorphism $\gamma: \mathbb{S}^2 \to \mathbb{S}^2$ is represented as a weighted sum of spherical harmonic bases, and the rotation $O \in SO(3)$ is solved using SVD. These two are optimized alternately until convergence; the entire process freezes the D-SNS weights and only optimizes the diffeomorphism parameters.
    - **Design Motivation**: SRNF linearizes the non-linear elastic metric, making registration optimization highly efficient; spherical harmonic bases naturally ensure the smoothness of the diffeomorphism.

3. **Temporal Registration (SRVF Space)**:

    - **Function**: Aligns the deformation speed differences between two 4D surfaces.
    - **Mechanism**: First uses PCA to reduce the 4D surfaces to low-dimensional curves, then maps them to the Square Root Velocity Fields (SRVF) space. The temporal diffeomorphism $\zeta: [0,1] \to [0,1]$ is implemented by a small MLP (2 residual blocks, 32 neurons) with its output constrained to $[0,1]$ using Sigmoid. Monotonicity is enforced via a regularization term $L_M = \int_0^1 \max(0, -\partial\zeta/\partial t)$.
    - **Design Motivation**: Performing temporal registration directly on 4D surfaces involves excessively high dimensionality and a non-linear metric. The dual simplification of PCA dimensionality reduction combined with SRVF mapping transforms the problem into solving elastic curve registration in a low-dimensional Euclidean space.

### Loss & Training
- D-SNS representation learning: MSE loss, trained for 50K epochs, with surface points randomly sampled each epoch.
- Spatial registration: $\mathbb{L}^2$ distance in the SRNF space, 500 iterations.
- Temporal registration: $L = \|q_1 - q_2 \circ \zeta\|^2 + \lambda L_M$, trained for 3000 epochs, updating temporal sampling points every 200 epochs.
- Mean computation: Karcher mean is jointly optimized in the SRVF space via Eqn.15.

## Key Experimental Results

### Main Results: Representation Accuracy

| Dataset | Mean Error ($\times 10^{-6}$) | Median Error ($\times 10^{-6}$) | Std. Dev. ($\times 10^{-6}$) |
|--------|------|------|------|
| DFAUST | 1.60 | 0.52 | 1.52 |
| CAPE | 1.51 | 0.68 | 1.27 |
| COMA | 2.29 | 1.68 | 1.66 |
| VOCA | 0.89 | 0.51 | 0.79 |

### Spatio-Temporal Registration Comparison (Geodesic Distance, smaller is better)

| Method | CAPE | DFAUST | COMA | VOCA |
|------|------|--------|------|------|
| 4D Atlas (Mean/Std/Med) | 1.29/0.48/1.40 | 2.38/1.12/1.87 | 0.06/0.02/0.06 | 0.50/0.11/0.41 |
| Ours (Mean/Std/Med) | 0.36/0.17/0.32 | 0.77/0.20/0.72 | 0.03/0.02/0.02 | 0.10/0.03/0.10 |

### Key Findings
- D-SNS representation error is on the level of $10^{-6}$, with point-wise error $<0.01$, faithfully reconstructing complex 4D surfaces.
- D-SNS trained with only 30 temporal samples can still perform high-quality interpolation of missing frames.
- When the temporal sampling drops from 50 to 25, the registration error of 4D Atlas increases significantly, whereas ours remains stable—demonstrating that the continuous representation is insensitive to sampling density.
- Across all four datasets, the geodesic distances of the proposed spatio-temporal registration are significantly lower than those of 4D Atlas.

## Highlights & Insights
- **Clever utilization of SRNF/SRVF**: Linearizing non-linear elastic metrics in different spaces transforms spatial and temporal registrations into optimization problems in Euclidean spaces. This philosophy of "finding an appropriate representation space to simplify the problem" is classic.
- **Advantages of continuous representation**: D-SNS allows all differential quantities to be calculated analytically, ensuring registration does not rely on discretization resolution. This provides strong evidence for the "continuous first, analysis later" paradigm.
- **Transferability**: The combination of MLPs as continuous function approximators and Riemannian elastic metric space analysis can be transferred to any task requiring statistical analysis of deforming objects, such as longitudinal organ analysis in medical imaging.

## Limitations & Future Work
- **Only supports genus-0 surfaces**: Relies on spherical parameterization, preventing it from handling objects with holes or high genus.
- **Computational efficiency**: Each 4D surface requires training an individual D-SNS (2-3 hours), which is costly when analyzing large-scale datasets. The authors suggest exploring conditional representations similar to DeepSDF in the future.
- **Texture/appearance not considered**: Only handles geometric shapes, without involving appearance information.
- **Dataset limitations**: All evaluation datasets consist of pre-registered triangular meshes; the capability of direct modeling from raw scan data is not validated.

## Related Work & Insights
- **vs 4D Atlas**: 4D Atlas operates on discrete signals, and its registration accuracy depends on temporal sampling density. This work operates on continuous functions, is more robust to sparse sampling, and comprehensively outperforms it across four datasets.
- **vs SMPL/SMPL-X**: SMPL explicitly models joint motion using skeletal joints and is a category-specific model. The proposed method is a general genus-0 surface analysis framework that is category-agnostic.
- **vs Neural Surface Maps**: Neural Surface Maps by Morreale et al. uses MLPs to represent static surface mappings. This study generalizes it to dynamic 4D surfaces and introduces a complete statistical analysis framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of neural representation with Riemannian statistical shape analysis shows some novelty, but each component (MLP fitting, SRNF/SRVF) relies on existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative and qualitative evaluations across four datasets are relatively comprehensive, though comparison is only conducted against a single method, 4D Atlas.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, but the full paper is long (22 pages), and the information density could be improved.
- Value: ⭐⭐⭐⭐ Provides a solid continuous framework for 4D shape analysis, though the application scenarios are somewhat niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 4Deform: Neural Surface Deformation for Robust Shape Interpolation](4deform_neural_surface_deformation_for_robust_shape_interpolation.md)
- [\[CVPR 2025\] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes](volumetric_surfaces_representing_fuzzy_geometries_with_layered_meshes.md)
- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)
- [\[CVPR 2025\] 3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping](3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)

</div>

<!-- RELATED:END -->
