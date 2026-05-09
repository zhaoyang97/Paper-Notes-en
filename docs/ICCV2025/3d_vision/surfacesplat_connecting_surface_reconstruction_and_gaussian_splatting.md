---
title: >-
  [Paper Note] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting
description: >-
  [ICCV 2025][3D Vision][Surface Reconstruction] SurfaceSplat proposes a hybrid framework that establishes bidirectional connections between SDF (Signed Distance Function) and 3D Gaussian Splatting (3DGS): the SDF provides coarse geometry to enhance 3DGS rendering quality, while novel-view images rendered by 3DGS are in turn used to refine SDF surface reconstruction accuracy. The method achieves state-of-the-art performance on both surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets.
tags:
  - ICCV 2025
  - 3D Vision
  - Surface Reconstruction
  - Gaussian Splatting
  - SDF
  - Sparse-View
  - Novel View Synthesis
date: 2026-05-08
content_hash: 0f0e7a4ba6991f89
---

# SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2507.15602](https://arxiv.org/abs/2507.15602)
**Code**: [https://github.com/aim-uofa/SurfaceSplat](https://github.com/aim-uofa/SurfaceSplat)
**Area**: 3D Vision
**Keywords**: Surface Reconstruction, Gaussian Splatting, SDF, Sparse-View, Novel View Synthesis

## TL;DR

SurfaceSplat proposes a hybrid framework that establishes bidirectional connections between SDF (Signed Distance Function) and 3D Gaussian Splatting (3DGS): the SDF provides coarse geometry to enhance 3DGS rendering quality, while novel-view images rendered by 3DGS are in turn used to refine SDF surface reconstruction accuracy. The method achieves state-of-the-art performance on both surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets.

## Background & Motivation

**Background**: Surface reconstruction and novel view synthesis from sparse-view images are core challenges in 3D vision. Two dominant technical paradigms exist: SDF-based neural implicit surface methods (e.g., NeuS, VolRecon), which learn signed distance fields to recover geometry, and 3D Gaussian Splatting (3DGS)-based methods, which optimize discrete 3D Gaussian primitives for fast, high-quality rendering.

**Limitations of Prior Work**: SDF methods excel at capturing globally consistent geometry but struggle with fine-grained details, as the resolution of implicit representations is limited by network capacity and tends to over-smooth under sparse-view settings. 3DGS methods offer superior rendering quality and speed, but their discrete point-cloud representation lacks global geometric constraints, leading to floating Gaussians (floaters) and inconsistent surfaces.

**Key Challenge**: The global geometric consistency of SDF and the local detail rendering capability of 3DGS are difficult to obtain simultaneously. Each representation has its own strengths, yet prior methods employ only one, or use one solely to initialize the other, without establishing a bidirectional co-optimization relationship.

**Goal**: Design a unified framework in which SDF and 3DGS mutually reinforce each other—SDF provides geometric priors for 3DGS, while 3DGS supplies multi-view supervision signals for SDF.

**Key Insight**: The authors observe that the strengths and weaknesses of SDF and 3DGS are complementary: SDF excels at global geometry but lacks detail, while 3DGS excels at detail but lacks global consistency. Enabling the two representations to mutually "teach" each other can improve both aspects simultaneously.

**Core Idea**: Establish a bidirectional information flow between SDF and 3DGS—using the coarse geometry from SDF to constrain the Gaussian distribution in 3DGS, while using high-quality novel-view images rendered by 3DGS as additional supervision for SDF optimization.

## Method

### Overall Architecture

SurfaceSplat takes sparse-view images (with camera parameters) as input and produces high-quality surface meshes and novel-view rendered images. The system comprises two branches optimized in parallel: an SDF branch (based on neural implicit surfaces) and a 3DGS branch (based on 3D Gaussian Splatting). The two branches are bidirectionally coupled through carefully designed information-passing mechanisms, mutually enhancing each other throughout joint optimization.

### Key Designs

1. **SDF→3DGS Geometry Enhancement**

   - **Function**: Leverages global geometric knowledge from SDF to improve the Gaussian distribution in 3DGS.
   - **Mechanism**: Extracts the zero-level set from the SDF network and uses mesh vertex positions and surface normals to initialize and constrain the 3DGS Gaussian primitives. Specifically, surface normals extracted from the SDF are used to regularize the orientation of Gaussians, and surface positions constrain the spatial distribution of Gaussian centers. This effectively reduces floaters commonly observed in 3DGS and causes Gaussian primitives to adhere more closely to the true surface.
   - **Design Motivation**: Floaters in 3DGS under sparse-view settings fundamentally arise from the absence of global geometric constraints. SDF naturally provides surface priors, making it intuitive and efficient to use them to constrain the Gaussian distribution.

2. **3DGS→SDF Detail Refinement**

   - **Function**: Uses novel-view images rendered by 3DGS to provide additional supervision signals for SDF.
   - **Mechanism**: 3DGS can efficiently render images from arbitrary viewpoints. During training, additional "virtual-view" images are rendered by 3DGS and used as pseudo-label supervision for SDF optimization. This effectively uses 3DGS's rendering capability to "augment" SDF training data—supplementing the sparse real input views with rendered images from additional viewpoints. To avoid introducing rendering artifacts, a confidence-weighted strategy is adopted so that only high-confidence rendered pixels contribute to SDF supervision.
   - **Design Motivation**: A key reason for poor SDF performance under sparse views is insufficient training data. As an effective interpolator, 3DGS can synthesize high-quality novel-view images between existing views, compensating for the data scarcity faced by SDF training.

3. **Joint Optimization Strategy**

   - **Function**: Coordinates the training process of the two branches.
   - **Mechanism**: An alternating optimization scheme is adopted—the SDF branch is first pre-trained to obtain initial geometry, which is then used to initialize 3DGS; thereafter, the two branches are optimized alternately while passing information to each other. The SDF→3DGS geometric constraints are updated at fixed intervals, while 3DGS-rendered images continuously supervise SDF. To prevent error propagation between branches, a progressive coupling strategy is introduced: the two branches operate relatively independently in early stages, with coupling strength gradually increased over time.
   - **Design Motivation**: Direct end-to-end joint training tends to cause training instability, as signals exchanged between two unconverged branches are highly noisy. Progressive coupling allows each branch to first acquire basic competence independently before gradually establishing collaboration.

### Loss & Training

The total loss comprises the SDF branch loss and the 3DGS branch loss. The SDF branch employs RGB rendering loss, depth regularization, and eikonal regularization; the 3DGS branch uses standard photometric loss and SSIM loss. The bidirectional information transfer is realized through an additional normal consistency loss $L_{normal}$ and a virtual-view supervision loss $L_{pseudo}$ with confidence weighting:

$$L_{pseudo} = \sum_i w_i \|I_{SDF}^i - I_{3DGS}^i\|_1$$

## Key Experimental Results

### Main Results

Surface reconstruction quality (Chamfer Distance, mm) and novel view synthesis quality (PSNR) on the DTU dataset (3 input views):

| Method | CD↓ (mm) | PSNR↑ | SSIM↑ | Type |
|--------|----------|-------|-------|------|
| SparseNeuS | 1.40 | 23.12 | 0.872 | SDF |
| VolRecon | 1.20 | 23.58 | 0.881 | SDF |
| C2F2NeuS | 1.13 | 24.01 | 0.889 | SDF |
| DNGaussian | - | 24.68 | 0.899 | 3DGS |
| CoR-GS | - | 24.32 | 0.895 | 3DGS |
| SurfaceSplat | **0.98** | **25.21** | **0.912** | Hybrid |

Results on the MobileBrick dataset:

| Method | CD↓ (mm) | F-score↑ | PSNR↑ |
|--------|----------|----------|-------|
| NeuS | 2.85 | 0.62 | 22.34 |
| 3DGS | - | - | 24.89 |
| SurfaceSplat | **2.12** | **0.74** | **25.67** |

### Ablation Study

| Configuration | CD↓ (mm) | PSNR↑ | Note |
|---------------|----------|-------|------|
| Full model | 0.98 | 25.21 | Complete model |
| w/o SDF→3DGS | 1.08 | 24.53 | More floaters in 3DGS without geometry enhancement |
| w/o 3DGS→SDF | 1.21 | 25.18 | Significant SDF accuracy drop without virtual-view supervision |
| w/o progressive coupling | 1.15 | 24.87 | Direct strong coupling causes training instability |
| SDF branch only | 1.35 | 23.45 | Pure SDF baseline |
| 3DGS branch only | - | 24.68 | Pure 3DGS baseline, no surface reconstruction |

### Key Findings

- The two directions of bidirectional information flow contribute differently: 3DGS→SDF virtual-view supervision contributes more to surface reconstruction quality (CD degrades from 0.98 to 1.21 when removed), while SDF→3DGS geometry enhancement contributes more to rendering quality (PSNR drops from 25.21 to 24.53 when removed).
- The progressive coupling strategy is critical for training stability; direct strong coupling causes early-stage errors to propagate between branches.
- Under more extreme sparse-view settings (e.g., 2 input views), SurfaceSplat demonstrates a larger advantage over pure SDF or pure 3DGS methods, as the complementary effect becomes more pronounced when data is scarcer.

## Highlights & Insights

- **Elegant Bidirectional Complementary Design**: The idea of having two representations mutually "teach" each other is both concise and effective. SDF teaches 3DGS geometry; 3DGS teaches SDF fine details, forming a virtuous cycle. This complementary design paradigm is transferable to other tasks where complementary representations exist.
- **Virtual-View Augmentation Strategy**: Using 3DGS-rendered novel views to augment SDF training data constitutes a low-cost data augmentation approach. The key technique is confidence-weighted filtering of low-quality renderings to avoid introducing noise.
- **Progressive Coupling**: This strategy addresses the engineering challenge of early error propagation in dual-branch systems. The "independence first, collaboration later" training paradigm has broad reference value in multi-task learning.

## Limitations & Future Work

- Training time is longer than single-representation methods due to the need to optimize two branches simultaneously with information exchange.
- The method still depends on accurate camera parameters; although robust to sparse views, inaccurate intrinsic or extrinsic calibration may cause both branches to propagate erroneous geometry.
- Validation is currently limited to object-level datasets (DTU, MobileBrick); applicability to large-scale scenes remains uncertain.
- The virtual-view selection strategy for 3DGS→SDF is relatively simple (random sampling); more intelligent view selection may further improve performance.

## Related Work & Insights

- **vs. NeuS/VolRecon**: Pure SDF methods achieve global consistency under sparse views but lack fine detail. SurfaceSplat compensates by introducing 3DGS rendering capability to supply additional detail supervision.
- **vs. DNGaussian/CoR-GS**: Pure 3DGS methods yield high rendering quality but lack surface geometry. SurfaceSplat resolves floater issues by leveraging SDF-based geometric constraints.
- **vs. 2DGS/SuGaR**: These methods attempt to incorporate surface constraints within 3DGS but remain single-representation approaches. SurfaceSplat's dual-representation design is more flexible, enabling simultaneous output of high-quality meshes and renderings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The bidirectional complementary design combining SDF and 3DGS is novel, though combining the two representations is not an entirely new idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ DTU and MobileBrick are standard benchmarks; ablation studies are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and method description is detailed.
- **Value**: ⭐⭐⭐⭐ Simultaneously improving reconstruction and rendering quality offers practical value in sparse-view scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](../../AAAI2026/3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](7dgs_unified_spatial-temporal-angular_gaussian_splatting.md)
- [\[ICCV 2025\] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](../../AAAI2026/3d_vision/meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)

<!-- RELATED:END -->
