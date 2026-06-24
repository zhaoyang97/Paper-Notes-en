---
title: >-
  [Paper Note] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting
description: >-
  [ICCV 2025][3D Vision][Gaze Redirection] This paper proposes GazeGaussian, the first high-fidelity gaze redirection method based on 3D Gaussian Splatting (3DGS). By employing a dual-stream 3DGS model to separately represent the facial and eye regions, the method introduces an explicit Gaussian eyeball rotation representation and an expression-guided neural renderer (EGNR), achieving state-of-the-art performance in gaze accuracy, synthesis quality, and rendering speed.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Gaze Redirection"
  - "3D Gaussian Splatting"
  - "Head Avatar Synthesis"
  - "Dual-Stream Model"
  - "Eyeball Rotation Representation"
date: 2026-05-08
content_hash: 53a782187ffcf6d6
---

# GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2411.12981](https://arxiv.org/abs/2411.12981)  
**Code**: [GitHub](https://ucwxb.github.io/GazeGaussian)  
**Area**: 3D Vision
**Keywords**: Gaze Redirection, 3D Gaussian Splatting, Head Avatar Synthesis, Dual-Stream Model, Eyeball Rotation Representation

## TL;DR

This paper proposes GazeGaussian, the first high-fidelity gaze redirection method based on 3D Gaussian Splatting (3DGS). By employing a dual-stream 3DGS model to separately represent the facial and eye regions, the method introduces an explicit Gaussian eyeball rotation representation and an expression-guided neural renderer (EGNR), achieving state-of-the-art performance in gaze accuracy, synthesis quality, and rendering speed.

## Background & Motivation

Gaze estimation is a fundamental task in computer vision; however, existing estimators suffer from poor generalization on out-of-distribution data. Gaze redirection addresses this by manipulating the gaze direction of an input image toward a target direction to generate augmented data, thereby improving the generalizability of gaze estimators.

**Limitations of Prior Work**:

**2D methods** (e.g., STED): Formulate gaze redirection as 2D image manipulation, ignoring the inherently 3D nature of head and gaze control, resulting in poor spatial consistency and limited synthesis fidelity.

**NeRF-based methods** (e.g., GazeNeRF): Incur high computational overhead and low rendering efficiency. Gaze direction is altered by implicitly rotating feature maps, which lacks precise controllability.

**3DGS-based head methods** (e.g., Gaussian Head Avatar): Neglect precise gaze direction control and cannot generalize across different subjects.

**Core Motivation**: The unstructured nature of 3DGS is inherently well-suited for explicit rigid rotation control of the eyeball. However, two key challenges must be addressed: (a) how to decouple facial animation from gaze motion, and (b) how to achieve generalization across different subjects.

## Method

### Overall Architecture

GazeGaussian consists of three core components:
1. Initialization of a dual-stream 3DGS (face + eye) using a pretrained neutral mesh.
2. Transformation of canonical Gaussians to the target space via a facial deformation field and an eyeball rotation field.
3. Generation of the final gaze-redirected image through the Expression-Guided Neural Renderer (EGNR).

Preprocessing includes background removal, gaze direction normalization, and face tracking to obtain per-frame identity/expression codes and camera poses.

### Key Designs

1. **Dual-Stream Gaussian Representation and Face Deformation Branch**:

   A facial Gaussian $\{\mu_0^f, z_0^f, R_0^f, S_0^f, \alpha_0^f\}$ is constructed in canonical space, where $z_0^f \in \mathbb{R}^{128}$ denotes the per-point feature vector.

   A key innovation is the distance-based influence weighting mechanism:
    - The minimum distance $d$ from each Gaussian center $\mu$ to 3D facial landmarks is computed.
    - Regions near landmarks ($d < d_1 = 0.15$) are primarily influenced by the expression code $\tau$, with $\lambda_\tau = 1$.
    - Distant regions ($d > d_2 = 0.25$) are primarily influenced by the head pose $\gamma$, with $\lambda_\tau = 0$.
    - Transition regions are interpolated smoothly: $\lambda_\tau = (d_2 - d)/(d_2 - d_1)$.

   Deformation is realized via an MLP: $\mu^f = \mu_0^f + \lambda_\tau E_\mu^f(\mu_0^f, \tau) + \lambda_\gamma P_\mu^f(\mu_0^f, \gamma)$

2. **Gaussian Eye Rotation Representation**:

   Unlike the face branch, the scaling of the eye branch is constrained to be isotropic, i.e., $S_0^e \in \mathbb{R}^{N \times 1}$, consistent with the rotational characteristics of the eyeball.

   The core design first rotates the eye Gaussians in canonical space and then incorporates the expression code to produce deformation offsets. Due to noise in gaze labels, two independent MLPs predict rotation offsets:

   $\mu^e = E_\mu^e(\mu_0^e, \tau) + G_\mu^e(\mu_0^e, \varphi) \mu_0^e$

   where $\varphi$ is the normalized gaze direction. This explicit rotation approach is more precise than GazeNeRF's implicit feature map rotation, fully leveraging the controllability of 3DGS.

3. **Expression-Guided Neural Renderer (EGNR)**:

   To address cross-subject generalization, the expression latent code $\tau$ is injected into the bottleneck features of a UNet-based renderer via slice cross-attention:

   $z_b' = z_b + z_b \cdot \text{Attn}(q = \tau, k = z_b, v = z_b)$

   This enables the renderer to be conditioned on subject-specific information, producing more realistic and personalized facial details.

### Loss & Training

**Image Synthesis Loss**: Rendered images and feature maps for the face, eye, and head regions are supervised separately:

$$\mathcal{L}_\mathcal{I}^e = \|\mathcal{I}_{gt} - \mathcal{I}_e\|_1 + \lambda_{SSIM}(1 - SSIM(\mathcal{I}_{gt}, \mathcal{I}_e)) + \lambda_{VGG} VGG(\mathcal{I}_{gt}, \mathcal{I}_e)$$

where $\lambda_{SSIM} = \lambda_{VGG} = 0.1$. The total image loss comprises six terms (three rendered images plus the RGB channels of three feature maps).

**Gaze Redirection Loss**: A pretrained gaze estimator is used to compute the angular error:

$$\mathcal{L}_\mathcal{G} = \mathcal{E}_{ang}(\psi^g(\mathcal{I}_h), \psi^g(\mathcal{I}_{gt}))$$

The final loss is $\mathcal{L} = 1.0 \cdot \mathcal{L}_\mathcal{I} + 0.1 \cdot \mathcal{L}_\mathcal{G}$.

## Key Experimental Results

### Main Results (Intra-dataset Evaluation on ETH-XGaze)

| Method | Gaze↓ | Head Pose↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FID↓ | ID↑ | FPS↑ |
|------|-------|-----------|-------|-------|--------|------|-----|------|
| STED | 16.217 | 13.153 | 0.726 | 17.530 | 0.300 | 115.020 | 24.347 | 18 |
| HeadNeRF | 12.117 | 4.275 | 0.720 | 15.298 | 0.294 | 69.487 | 46.126 | 35 |
| GazeNeRF | 6.944 | 3.470 | 0.733 | 15.453 | 0.291 | 81.816 | 45.207 | 46 |
| Gaussian Head Avatar | 30.963 | 13.563 | 0.638 | 12.108 | 0.359 | 74.560 | 27.272 | 91 |
| **GazeGaussian** | **6.622** | **2.128** | **0.823** | **18.734** | **0.216** | **41.972** | **67.749** | **74** |

GazeGaussian slightly outperforms GazeNeRF in gaze accuracy (6.622° vs. 6.944°), improves head pose accuracy by 38% (2.128° vs. 3.470°), achieves comprehensively superior image quality (PSNR gain of 3.28 dB), and attains 74 FPS (1.6× that of GazeNeRF).

### Ablation Study

| Two-stream | Gaussian Eye Rep. | Expression-Guided | Gaze↓ | Head↓ | SSIM↑ | PSNR↑ | LPIPS↓ | FID↓ | ID↑ |
|:-:|:-:|:-:|-------|-------|-------|-------|--------|------|------|
| ✓ | | | 13.651 | 2.981 | 0.753 | 16.376 | 0.272 | 55.481 | 38.941 |
| ✓ | ✓ | | 13.489 | 3.149 | 0.751 | 16.365 | 0.274 | 54.327 | 41.521 |
| ✓ | | ✓ | 8.883 | 2.635 | - | - | - | - | - |
| ✓ | ✓ | ✓ | **6.622** | **2.128** | **0.823** | **18.734** | **0.216** | **41.972** | **67.749** |

All three components are indispensable: the dual-stream model provides the foundation for face–eye decoupling; the Gaussian eye rotation representation reduces gaze error from 13.65° to 6.62°; and the expression-guided renderer substantially improves identity preservation (ID: 38.9 → 67.7).

### Cross-Dataset Generalization

| Method | Columbia Gaze↓ | Columbia ID↑ | MPII Gaze↓ | MPII ID↑ | GazeCapture Gaze↓ | GazeCapture ID↑ |
|------|---------------|-------------|-----------|---------|-------------------|----------------|
| GazeNeRF | 9.464 | 23.157 | 14.933 | 30.981 | 10.463 | 19.025 |
| GazeGaussian | **7.415** | **59.788** | **10.943** | **41.505** | **9.752** | **44.007** |

GazeGaussian achieves the best gaze accuracy and identity preservation across all three cross-dataset evaluations.

### Key Findings

1. Although Gaussian Head Avatar achieves fast rendering (91 FPS), the lack of a gaze decoupling mechanism causes complete failure in gaze redirection (30.96° error).
2. GazeNeRF's implicit feature map rotation performs poorly under extreme gaze directions.
3. Explicit Gaussian eyeball rotation significantly outperforms implicit methods in both accuracy and controllability.

## Highlights & Insights

- **First 3DGS-based gaze redirection method**: Explicit eyeball rotation control is realized by exploiting the unstructured nature of 3DGS, yielding a conceptually clean and effective design.
- **Elegant dual-stream decoupling**: The face and eye branches adopt distinct deformation/rotation strategies that conform to their respective physical motion characteristics.
- **Distance-based weighting mechanism**: Smooth interpolation of expression and pose control based on distance to landmarks avoids hard boundaries.
- **Expression-guided renderer improves generalization**: Cross-attention injection of subject-specific information addresses the single-subject limitation common to 3DGS-based methods.

## Limitations & Future Work

- Training still requires 14.4K images, leaving room for improvement in data efficiency.
- Constraining scaling to isotropic in the eyeball rotation representation may limit the modeling accuracy of non-spherical eye structures.
- More complex eye motions such as eye closure and blinking are not considered.
- Noise in gaze labels is compensated via MLP-predicted offsets, but the fundamental issue of label quality remains unresolved.

## Related Work & Insights

- Gaussian Head Avatar provides the foundational framework for 3DGS-based head modeling; this work extends it with gaze control capability.
- GazeNeRF establishes the dual-stream face–eye decoupling paradigm; this work transfers it from NeRF to 3DGS and adds explicit control.
- The cross-attention design of the expression-guided renderer is generalizable to other conditional rendering tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First 3DGS-based gaze redirection method with a distinctive explicit eyeball rotation design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation on four datasets with complete ablations, cross-dataset generalization, and downstream task verification.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and intuitive visual comparisons.
- **Value**: ⭐⭐⭐⭐ Highly practical with clear application scenarios in gaze redirection and data augmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [\[AAAI 2026\] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image](../../AAAI2026/3d_vision/rtgaze_real-time_3d-aware_gaze_redirection_from_a_single_image.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[NeurIPS 2025\] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](../../NeurIPS2025/3d_vision/planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)
- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)

</div>

<!-- RELATED:END -->
