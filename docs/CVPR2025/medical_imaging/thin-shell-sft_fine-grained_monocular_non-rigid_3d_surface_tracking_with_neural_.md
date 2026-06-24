---
title: >-
  [Paper Note] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields
description: >-
  [CVPR 2025][Medical Imaging][Non-rigid surface tracking] Thin-Shell-SfT proposes a monocular non-rigid 3D surface tracking method based on continuous neural deformation fields and Kirchhoff-Love thin-shell physical priors, combined with surface-induced 3D Gaussian splatting for differentiable rendering, achieving unprecedented accuracy in fine-grained wrinkle reconstruction.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Non-rigid surface tracking"
  - "thin-shell mechanics"
  - "neural deformation fields"
  - "Gaussian splatting"
  - "monocular reconstruction"
date: 2026-05-08
content_hash: f685e8cd545239b3
---

# Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields

**Conference**: CVPR 2025  
**arXiv**: [2503.19976](https://arxiv.org/abs/2503.19976)  
**Code**: [https://4dqv.mpi-inf.mpg.de/ThinShellSfT](https://4dqv.mpi-inf.mpg.de/ThinShellSfT) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Non-rigid surface tracking, thin-shell mechanics, neural deformation fields, Gaussian splatting, monocular reconstruction

## TL;DR

Thin-Shell-SfT proposes a monocular non-rigid 3D surface tracking method based on continuous neural deformation fields and Kirchhoff-Love thin-shell physical priors, combined with surface-induced 3D Gaussian splatting for differentiable rendering, achieving unprecedented accuracy in fine-grained wrinkle reconstruction.

## Background & Motivation

**Background**: Reconstructing the 3D shape of highly deformable surfaces (such as cloth) from monocular RGB video is a highly challenging and ill-posed problem. Recently, Shape-from-Template (SfT) methods have made progress, particularly physics-based methods (such as $\boldsymbol{\phi}$-SfT) that achieve SOTA performance through differentiable physical simulators and differentiable rendering.

**Limitations of Prior Work**: Even SOTA $\boldsymbol{\phi}$-SfT has severe limitations: (1) Its underlying representation relies on discrete polygonal meshes, making it difficult to balance resolution details and efficiency (about 300 vertices can only capture coarse deformations); (2) FEM simulators exhibit inconsistent behavior across different resolutions, preventing the adoption of coarse-to-fine strategies; (3) Frame-by-frame optimization leads to error accumulation and local minima issues; (4) Mesh-based differentiable renderers provide poor gradient quality and do not support dynamic remeshing.

**Key Challenge**: The contradiction between discrete mesh representations and continuous surface deformations—fine-grained wrinkles require extremely high-resolution meshes, which brings unacceptable computational and memory overhead. Moreover, the discretization of FEM simulators is inconsistent across resolutions, making adaptive precision adjustment impossible.

**Goal**: Replace the discrete mesh with a continuous adaptive surface representation, combined with a continuous physical prior, to achieve fine-grained cloth wrinkle and fold reconstruction.

**Key Insight**: Neural implicit fields can represent continuous surfaces and naturally support queries at arbitrary resolutions; the Kirchhoff-Love thin-shell model can impose physical constraints in the continuous domain (rather than on mesh vertices); 3D Gaussian splatting provides better gradients for differentiable rendering compared to triangular meshes.

**Core Idea**: Represent the continuous deformation of the surface using a spatio-temporal neural deformation field (NDF), impose a continuous Kirchhoff-Love thin-shell internal energy minimization constraint on the deformation field, and establish photometric alignment with the input images using surface-induced 3D Gaussian splatting.

## Method

### Overall Architecture

Given a template surface $\mathbf{S}_1$ (the first frame) and a monocular video sequence $\{\mathbf{I}_t\}$. First, a continuous representation of the template $\bar{\mathbf{x}}(\boldsymbol{\xi})$ is fitted using an NRF (Neural Reference Field). Then, a neural deformation field (NDF) $\mathbf{u}(\boldsymbol{\xi}, t)$ is optimized such that the deformed surface $\mathbf{x}(\boldsymbol{\xi}, t) = \bar{\mathbf{x}}(\boldsymbol{\xi}) + \mathbf{u}(\boldsymbol{\xi}, t)$, when rendered via Gaussian splatting, matches the input images while satisfying thin-shell physical constraints. The output is a continuous spatio-temporal surface sequence that can be queried at arbitrary resolutions.

### Key Designs

1. **Continuous Neural Deformation Field (NDF)**:

    - **Function**: Represents the 3D displacement of the surface at any parametric domain point and any timestep.
    - **Mechanism**: Utilizes a SIREN MLP (sinusoidal activation function, $\omega=30$) to implement $\mathcal{F}(\boldsymbol{\xi}, t; \Theta)$, taking parametric coordinates and time as inputs and outputting deformation offsets. A key design is momentum conservation: $\mathbf{u}(\boldsymbol{\xi}, t) = \lambda \mathbf{u}(\boldsymbol{\xi}, t-1) + \mathcal{F}(\boldsymbol{\xi}, t)$ ($\lambda=0.4$), allowing current deformation to continue along the direction of the previous frame's deformation. Global spatio-temporal joint optimization is performed across all frames (rather than random frames), and during optimization, gradients from later frames can flow back to update previous frames.
    - **Design Motivation**: A continuous representation ensures that: (1) inquiries can be made at arbitrary resolutions; (2) the high-frequency representation capability of SIREN can capture fine folds; (3) the global MLP provides a low-dimensional, smooth deformation space, yielding natural regularization. The momentum term encourages causally continuous motion.

2. **Continuous Kirchhoff-Love Thin-Shell Physical Prior**:

    - **Function**: Ensures physical plausibility of the deformation—preventing the surface from stretching, compressing, or bending unnaturally.
    - **Mechanism**: Models the surface as a Kirchhoff-Love thin shell, computing the non-linear membrane strain $\boldsymbol{\varepsilon}$ (in-plane stretching) and bending strain $\boldsymbol{\kappa}$ (curvature change) from the deformation gradient outputted by the NDF. The physical loss is the hyperelastic internal energy $\mathcal{L}_p = \frac{1}{2} \sum(D \boldsymbol{\varepsilon}^\top \mathbf{H} \boldsymbol{\varepsilon} + B \boldsymbol{\kappa}^\top \mathbf{H} \boldsymbol{\kappa}) \sqrt{\bar{a}}$, where $D$ and $B$ represent the in-plane and bending stiffnesses, respectively. In each iteration, $N_p=100$ points are randomly resampled in the parameter domain to evaluate the physical loss. A linear isotropic material is assumed ($E=5000$ Pa, $\nu=0.25$).
    - **Design Motivation**: Unlike the discrete FEM simulator in $\boldsymbol{\phi}$-SfT, the physical constraint here acts on a continuous surface—randomly resampling different points in each iteration achieves adaptive discretization. It does not require knowing external forces (their role is replaced by the photometric loss), only requiring the minimization of internal energy.

3. **Surface-Induced 3D Gaussian Splatting**:

    - **Function**: Differentiably renders the continuous surface into the image space, providing high-quality gradients to drive deformation optimization.
    - **Mechanism**: Samples approximately 90k Poisson disk points from the template mesh as Gaussian centers. Key constraints: (1) Gaussian positions are determined by the deformation output of the NDF; (2) the rotation matrix is fixed to the local coordinate system on the template (tangents + normal); (3) the scale along the normal direction is fixed to a minimal value $\epsilon=10^{-5}$; (4) colors, opacities, and tangential scales are learned only from the first frame and frozen for subsequent frames. Gaussians move along with surface points during deformation.
    - **Design Motivation**: In a monocular setting, multi-view information is unavailable. Learning Gaussian attributes from all frames, as in standard 3DGS, would lead to incorrect reconstructions due to deformation-texture-appearance ambiguity. "Locking" the Gaussians onto the surface (learning appearance only from the template, and only learning deformation in subsequent frames) eliminates this ambiguity.

### Loss & Training

The total loss is $\mathcal{L} = \lambda_d \mathcal{L}_d + \lambda_p \mathcal{L}_p$, where $\lambda_d=5$, $\lambda_p=1$. The data loss $\mathcal{L}_d$ contains the $\ell_1$ photometric loss and an optional silhouette loss. All frames are optimized jointly in each iteration (no random frame sampling). NRF pre-training takes about 2 minutes, while the core NDF training takes 30 minutes to 1 hour on an NVIDIA A100 GPU. A SIREN architecture with 5 hidden layers of 256 units is used.

## Key Experimental Results

### Main Results

Chamfer distance ($\times 10^4$) on the $\boldsymbol{\phi}$-SfT dataset:

| Method | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Mean |
|------|-----|-----|-----|-----|------|-----|-----|-----|-----|------|
| DDD | 2.95 | 1.69 | 3.80 | 25.73 | 10.46 | 6.97 | 15.64 | 7.61 | 11.77 | 10.87 |
| $\boldsymbol{\phi}$-SfT | 0.79 | 2.75 | 3.54 | 7.60 | 6.15 | 3.14 | 4.73 | 2.52 | 2.36 | 3.93 |
| **Ours** | 1.17 | **0.55** | **2.4** | **5.5** | 8.69 | **2.51** | **3.8** | **2.27** | 3.00 | **3.3** |

Normal consistency metrics: The proposed method achieves an $\ell_2$ normal error of 0.009 and a cosine normal error of 0.034, compared to $\boldsymbol{\phi}$-SfT's 0.013 and 0.041.

### Ablation Study

| Configuration | Mean Chamfer ($\times 10^4$) | Description |
|------|--------------------------|------|
| W/o physics prior | 34.25 | Severe stretching/shrinking of the surface |
| W/o surface-induced Gaussians | 14.0 | Ambiguity in Gaussian attribute learning |
| W/o fixed normal scale | 3.75 | Gaussians elongated along the normal |
| Full Model | **3.46** | Best performance with all components cooperating |

### Key Findings

- Thin-Shell-SfT achieves an overall average Chamfer distance of 3.3 (vs. 3.93 for $\boldsymbol{\phi}$-SfT), outperforming the SOTA on 6 out of 9 sequences.
- The quality of normal reconstruction is significantly improved (error reduced by ~30%), showing that fine-grained wrinkles are captured much better.
- The physical prior is the most critical component—without it, the Chamfer distance explodes by about 10 times (34.25 vs 3.46).
- Global joint optimization (vs. random frame optimization) is crucial to avoid local minima at folds.
- The runtime advantage is significant: 30 minutes to 1 hour (vs. several hours for $\boldsymbol{\phi}$-SfT, which can only handle around 50 frames).
- Comparisons with dynamic view synthesis methods (K-Planes, Deformable Gaussians) demonstrate that they fail to recover temporally consistent surface geometry in a monocular static camera setting.

## Highlights & Insights

- **The paradigm shift from discrete to continuous** is the core contribution. The combination of a continuous neural field and a continuous physical prior eliminates the difficulty of selecting a discrete mesh resolution, and adaptive discretization is achieved through resampling in each iteration—this is far more flexible than FEM methods with fixed mesh resolutions.
- **An ingenious insight of "replacing external forces with image loss."** In inverse problems, external forces are unknown, but the photometric loss can act as the external force—driving the surface deformation to match the observations, while the physical prior constrains the intrinsic behavior of the deformation. This separation is highly elegant.
- The design idea of surface-induced 3DGS—"locking" Gaussians onto the surface and restricting their degrees of freedom—can be transferred to any scenario requiring geometry reconstruction from monocular videos.

## Limitations & Future Work

- Material properties (Young's modulus, Poisson's ratio) need to be set manually; different cloth materials might require different parameters.
- Extreme self-collision (multi-layer overlapping folds) remains difficult; the physical prior currently does not handle contacts.
- Tracking untextured surfaces is an open challenge—the photometric loss degrades in the absence of texture gradients.
- The template assumption (requiring a complete 3D surface for the first frame) limits the generality of the method.
- Future work can treat material parameters as optimization variables to achieve end-to-end material estimation and shape tracking.

## Related Work & Insights

- **vs $\boldsymbol{\phi}$-SfT**: $\boldsymbol{\phi}$-SfT uses a discrete FEM simulator + mesh renderer, which is limited by low-resolution meshes of ~300 vertices. Thin-Shell-SfT uses a continuous neural field + continuous physics + Gaussian splatting, inherently breaking the resolution bottleneck.
- **vs Stotko et al.**: That method also uses physical priors but relies on a fixed-resolution proxy model. The resampling-per-iteration strategy in this paper provides adaptive physical constraint resolution.
- **vs NeuralClothSim**: NeuralClothSim is a forward simulator (predicting the equilibrium state given known forces and materials). This paper is the first to apply its ideas to the inverse problem (inferring deformation from images), which requires handling new challenges, such as unknown forces and a lack of lower bounds.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of continuous physical priors, neural deformation fields, and surface-induced 3DGS is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully evaluated on standard datasets with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ High technical depth and detailed mathematical derivations, but presents a high barrier to entry for readers without a background in thin-shell physics.
- Value: ⭐⭐⭐⭐⭐ Significantly pushes the boundaries of non-rigid 3D tracking; the paradigm of continuous representation may influence the entire field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections](crosssdf_3d_reconstruction_of_thin_structures_from_cross-sections.md)
- [\[ECCV 2024\] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration](../../ECCV2024/medical_imaging/textttnephi_neural_deformation_fields_for_approximately_diff.md)
- [\[CVPR 2025\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2025\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)
- [\[CVPR 2025\] CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](cholectrack20_a_multi-perspective_tracking_dataset_for_surgical_tools.md)

</div>

<!-- RELATED:END -->
