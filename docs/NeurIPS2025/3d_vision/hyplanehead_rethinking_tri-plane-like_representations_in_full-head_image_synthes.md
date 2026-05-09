---
title: >-
  [Paper Note] HyPlaneHead: Rethinking Tri-plane-like Representations in Full-Head Image Synthesis
description: >-
  [NeurIPS 2025][3D Vision][3D-aware GAN] This paper systematically analyzes three fundamental problems of tri-plane-like representations in 3D-aware head synthesis — mirror artifacts, non-uniform mapping, and feature penetration — and proposes a hybrid hy-plane representation (planar + spherical) combined with a unify-split strategy and near-equal-area warping, achieving state-of-the-art performance in full-head image synthesis.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D-aware GAN
  - tri-plane
  - full-head synthesis
  - feature entanglement
  - hybrid representation
date: 2026-05-08
content_hash: 62bb5e44915455c1
---

# HyPlaneHead: Rethinking Tri-plane-like Representations in Full-Head Image Synthesis

**Conference**: NeurIPS 2025
**arXiv**: [2509.16748](https://arxiv.org/abs/2509.16748)
**Code**: None
**Area**: 3D Vision
**Keywords**: 3D-aware GAN, tri-plane, full-head synthesis, feature entanglement, hybrid representation

## TL;DR

This paper systematically analyzes three fundamental problems of tri-plane-like representations in 3D-aware head synthesis — mirror artifacts, non-uniform mapping, and feature penetration — and proposes a hybrid hy-plane representation (planar + spherical) combined with a unify-split strategy and near-equal-area warping, achieving state-of-the-art performance in full-head image synthesis.

## Background & Motivation

**Background**: 3D-aware GANs (e.g., EG3D) encode 3D objects onto three orthogonal 2D feature planes using tri-plane representations, querying features via Cartesian coordinate projection. This approach balances efficiency and expressiveness and has been widely adopted for human head synthesis and 3D object modeling.

**Limitations of Prior Work**:
- **Mirror artifacts** (Tri-plane): Cartesian projection causes two 3D points symmetric about a feature plane to query identical features, producing ghost faces on the back of the head.
- **Non-uniform mapping** (SphereHead): Spherical coordinate $(\theta, \phi)$ mapping leads to sparse features near the equator and dense features near the poles, reducing feature map utilization and detail generation capability.
- **Feature penetration** (both): Convolutional networks generate features for different planes via different channels, yet the same UV location carries entirely different spatial meanings across planes, causing features from dominant planes to "penetrate" into others.

**Key Challenge**: Tri-plane excels at symmetric features but cannot handle asymmetric regions; spherical tri-plane resolves directionality but introduces non-uniform distribution and seam artifacts.

**Goal**: Simultaneously address all three problems: mirror artifacts, non-uniform feature distribution, and feature penetration.

**Key Insight**: Combine planar and spherical representations in a hybrid manner, leveraging the strengths of each.

**Core Idea**: Use planar features to capture symmetric structures and spherical features to capture directional structures, coupled with unify-split to eliminate inter-channel feature penetration and near-equal-area warping to optimize spherical feature distribution.

## Method

### Overall Architecture

HyPlaneHead is a 3D-aware GAN whose generator outputs a single-channel unified feature map, which is then spatially split into the individual planes of the hy-plane representation. Hy-plane is a hybrid combination of Cartesian feature planes (2 or 3) and spherical feature planes (1 or 2). Volume rendering is applied to generate head images, followed by a super-resolution module to produce high-resolution outputs.

### Key Designs

1. **Hy-Plane Representation**:

   - **Function**: Encodes 3D features using a hybrid of Cartesian planes and spherical planes.
   - **Design Motivation**: Planar representations excel at capturing symmetric features (e.g., bilateral ear symmetry), while spherical representations distinguish directional features (e.g., front face vs. back of the head).
   - **Mechanism**:
     - **Hy-plane (3+1)**: Three orthogonal Cartesian planes + one spherical plane; features from Cartesian and spherical projections are fused at query time.
     - **Hy-plane (2+2)**: Two orthogonal planes + two spherical planes with opposing pole orientations; spherical features are fused via a weighted function.
   - **Novelty**: Unlike PanoHead's tri-grid (which adds more parallel planes), hy-plane fundamentally introduces spherical surfaces to eliminate directional entanglement.

2. **Near-Equal-Area Warping**:

   - **Function**: Maps a square feature map onto the sphere to ensure uniform feature distribution.
   - **Design Motivation**: Direct $(\theta, \phi)$ mapping causes sparse equatorial coverage, dense polar coverage, and numerical discontinuities at $\phi = \pm\pi$.
   - **Mechanism**: A two-step transformation:
     1. Lambert Azimuthal Equal-Area (LAEA) projection: unfolds the sphere from the south pole onto a circular plane:
     $$(R, \Theta) = \left(2\cos\frac{1}{2}\phi,\ -\theta\right)$$
     2. Elliptical grid mapping: transforms the circle into a square:
     $$u = \frac{1}{2}\sqrt{2+x^2-y^2+2\sqrt{2}x} - \frac{1}{2}\sqrt{2+x^2-y^2-2\sqrt{2}x}$$
   - **Novelty**: LAEA consolidates seams and both poles into a single point directed toward the invisible underside of the head, completely eliminating seam artifacts.

3. **Unify-Split Strategy**:

   - **Function**: Replaces the multi-channel scheme (where different channels correspond to different planes) with a single-channel unified feature map that is spatially partitioned.
   - **Design Motivation**: In RGB images, the three channels share the same 2D spatial semantics (differing only in color channel); however, in tri-plane representations, different channels encode features with entirely different spatial orientations. Convolutional kernels compute all channels from the same input at each UV location, making it difficult to produce outputs with fundamentally different spatial meanings.
   - **Mechanism**:
     - The generator outputs a single large-channel feature map, which is spatially split into individual feature planes.
     - **Uniform split**: 2×2 equal partition.
     - **Area-bias split**: Allocates larger area to the spherical plane to enhance directional expressiveness.
   - **Novelty**: Completely eliminates inter-channel feature penetration, as each plane is physically separated in 2D space.

4. **Dual-Sphere Fusion (Hy-plane 2+2)**:

   - **Function**: Uses two spherical planes with opposing pole orientations to complement each other and resolve pole artifacts.
   - **Mechanism**: Features are fused with weights inversely proportional to the projection radius:
   $$w_a = (R_a^{\max} - R_a)^2, \quad f_{\text{sph}} = \frac{w_a f_a + w_b f_b}{w_a + w_b}$$
   - **Core Idea**: Central regions receive the highest weight (flattest in the feature map) while boundary regions receive the lowest (greatest distortion); the two spherical planes mutually compensate for each other's polar regions.

### Loss & Training

- Standard 3D-aware GAN losses are adopted (identical to EG3D).
- A view-image consistency loss (from SphereHead) is added to guide the discriminator to focus on pose-image alignment.
- An independent background generator is introduced so that the main generator focuses on the head region.
- Training is performed on 8× NVIDIA V100 GPUs with a batch size of 32 over 25M images.

## Key Experimental Results

### Main Results: FID Comparison for Full-Head Image Synthesis

| Representation | Unify-Split | Warping | FID↓ | FID-random↓ |
|---|---|---|---|---|
| Tri-plane (EG3D) | - | - | 9.22 | 11.23 |
| Tri-plane | evenly split | - | 8.86 | 11.52 |
| Spherical Tri-plane (SphereHead) | - | - | 8.64 | 10.71 |
| Spherical Tri-plane | evenly split | - | 8.36 | 10.42 |
| Tri-grid (PanoHead) | - | - | 8.77 | 10.66 |
| Hy-plane (3+1) | - | - | 8.54 | 10.66 |
| Hy-plane (3+1) | evenly split | - | 8.31 | 10.18 |
| Hy-plane (3+1) | evenly split | yes | 8.18 | 9.96 |
| **Hy-plane (3+1)** | **area-bias** | **yes** | **8.14** | **9.88** |
| **Hy-plane (2+2)** | **area-bias** | **yes** | **8.17** | **9.84** |

### Ablation Study

| Ablation Dimension | Finding |
|---|---|
| Introduction of spherical plane | Tri-plane → Hy-plane(3+1): FID 9.22→8.54, FID-random 11.23→10.66 |
| Unify-Split strategy | Reduces FID across all representations; increases FID-random for Tri-plane (due to lack of spherical disentanglement) |
| Near-equal-area warping | FID 8.31→8.18, FID-random 10.18→9.96 |
| Area-bias split | Further marginal improvement (FID 8.18→8.14) |
| Feature map resolution (256²→512²) | Negligible effect on Tri-plane/SphereHead, ruling out parameter count as a confound |
| Dual-sphere shared vs. separate branches | Sharing a single branch output for both spheres causes severe interference (FID 11.9/13.54) |

### Key Findings

- Unify-Split produces an interesting contradictory effect on Tri-plane: FID decreases but FID-random increases — eliminating penetration allows each plane to express its own features more fully, but the mirror artifact problem of Tri-plane consequently becomes more pronounced.
- Hy-plane (2+2) achieves slightly better FID-random than (3+1), as the dual spheres more effectively handle polar regions.
- Visualizations clearly show that secondary planes in Tri-plane and Spherical representations exhibit strong penetration of dominant-plane textures, while Unify-Split completely eliminates this phenomenon.

## Highlights & Insights

- **Thorough problem analysis**: The paper is the first to systematically identify and analyze the feature penetration problem, supported by clear visualizations.
- **Elegant solution**: The Unify-Split strategy requires no additional parameters yet completely resolves feature penetration.
- **Mathematical elegance of near-equal-area warping**: LAEA combined with elliptical grid mapping simultaneously addresses three issues — seams, pole singularities, and non-uniform distribution.
- **Complementary design philosophy**: Rather than simply replacing Tri-plane or SphereHead, the method synthesizes the strengths of both.

## Limitations & Future Work

- The current work focuses on human head synthesis; generalization to other 3D objects and scenes remains to be validated.
- kNN graph construction introduces additional engineering complexity.
- Hy-plane (3+1) still relies on the prior assumption that the south pole faces downward, limiting generality (the (2+2) variant addresses this).
- Training requires substantial computational resources: 8 GPUs and 25M training images.
- No comparison is made against recent 3D Gaussian Splatting or diffusion model-based methods.

## Related Work & Insights

- **EG3D**: The seminal work introducing tri-plane representations; this paper analyzes its three fundamental limitations.
- **SphereHead**: Resolves mirror artifacts via spherical coordinate systems but introduces non-uniform mapping as a new problem.
- **PanoHead**: Enhances tri-plane with tri-grid but does not fundamentally address the mirror issue.
- **Insights**: In 3D representation design, the root cause of problems often lies in the impedance mismatch between coordinate system choices and network architecture. Hybrid coordinate systems and physically separating feature channels with different spatial semantics constitute a powerful design paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ — Hybrid representation, Unify-Split, and near-equal-area warping each offer genuine novelty, though each contribution appears incremental in isolation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations and convincing visualizations, but user studies and additional quantitative metrics are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem analysis is thorough, figures are well-crafted, and the presentation is logically coherent.
- Value: ⭐⭐⭐⭐ — Directly valuable to the 3D-aware GAN community, though the overall performance margin may be limited.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] From Image to Video: An Empirical Study of Diffusion Representations](../../ICCV2025/3d_vision/from_image_to_video_an_empirical_study_of_diffusion_representations.md)
- [\[ICCV 2025\] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](../../ICCV2025/3d_vision/humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)
- [\[NeurIPS 2025\] 3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)
- [\[ICCV 2025\] PolarAnything: Diffusion-based Polarimetric Image Synthesis](../../ICCV2025/3d_vision/polaranything_diffusion-based_polarimetric_image_synthesis.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)

<!-- RELATED:END -->
