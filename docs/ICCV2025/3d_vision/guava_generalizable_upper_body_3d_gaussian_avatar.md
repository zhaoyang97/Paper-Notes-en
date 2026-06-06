---
title: >-
  [Paper Note] GUAVA: Generalizable Upper Body 3D Gaussian Avatar
description: >-
  [ICCV 2025][3D Vision][3D Gaussian] This paper presents GUAVA, the first framework for feed-forward reconstruction of animatable upper-body 3D Gaussian avatars from a single image. By combining template Gaussians and UV…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian"
  - "upper-body avatar"
  - "generalizable reconstruction"
  - "single-image reconstruction"
  - "expression driving"
  - "real-time rendering"
  - "SMPLX"
  - "FLAME"
date: 2026-05-08
content_hash: 19bb0d694bc8990a
---

# GUAVA: Generalizable Upper Body 3D Gaussian Avatar

**Conference**: ICCV 2025
**arXiv**: [2505.03351](https://arxiv.org/abs/2505.03351)  
**Code**: N/A (see project page)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian, upper-body avatar, generalizable reconstruction, single-image reconstruction, expression driving, real-time rendering, SMPLX, FLAME

## TL;DR

This paper presents GUAVA, the first framework for feed-forward reconstruction of animatable upper-body 3D Gaussian avatars from a single image. By combining template Gaussians and UV Gaussians in a canonical space representation, GUAVA supports rich facial expression and gesture driving, completing reconstruction in approximately 0.1 s with real-time rendering capability.

## Background & Motivation

### State of the Field

Creating realistic, expressive upper-body avatars is critical for applications in film, gaming, and virtual conferencing. Existing methods suffer from multiple limitations:

**Limitations of 3D methods:**

### Limitations of Prior Work

Per-identity training is required (GART, GaussianAvatar, ExAvatar), with optimization times ranging from minutes to hours per identity.

### Root Cause

Multi-view or monocular video inputs are required, imposing high data acquisition costs.

### Starting Point

Head reconstruction methods (e.g., GAGAvatar) lack body motion representation, while full-body methods neglect fine-grained facial expressions.

**Limitations of 2D diffusion-based methods:**

### Supplementary Notes

Generated quality is high but identity consistency is poor, especially under large pose changes.

### Supplementary Notes

Inference is slow, requiring multi-step denoising per frame.

### Supplementary Notes

Viewpoint cannot be flexibly controlled.

GUAVA occupies a unique niche as **the first generalizable upper-body avatar reconstruction method from a single image**, simultaneously addressing speed (~0.1 s reconstruction + 52 FPS rendering), quality (surpassing both 2D and 3D baselines), and expressiveness (facial expressions + hand gestures).

## Method

### Overall Architecture

The pipeline consists of four stages:
1. **EHM template model and tracking**: Obtaining accurate shape, expression, and pose parameters.
2. **Dual-branch reconstruction**: Template Gaussians + UV Gaussians to build upper-body Gaussians in canonical space.
3. **Animation and deformation**: Deforming canonical-space Gaussians to target poses using tracked parameters.
4. **Rendering and refinement**: Gaussian splatting renders a coarse feature map, which a StyleUNet refiner decodes into the final image.

### Key Designs

**EHM (Expressive Human Model)**: Addresses the limited facial expressiveness of SMPLX by replacing the SMPLX head with the FLAME model, aligning the two via eye joint displacement vectors. Tracking proceeds in two stages: coarse estimation with a pretrained model followed by fine optimization with 2D keypoint losses. Learnable joint offsets are introduced to improve alignment accuracy.

**Template Gaussians**:
- A pretrained DINOv2 extracts image features and a global identity embedding.
- EHM vertices are projected to screen space, and appearance features are sampled via bilinear interpolation.
- Each vertex also carries an optimizable base feature encoding unique semantic information.
- An MLP decoder concatenates all three to predict Gaussian attributes (rotation, scale, opacity, color features).
- Gaussian positions are set directly to vertex positions.

**UV Gaussians**: Address the limited vertex count of the template mesh and its inability to capture high-frequency details.
- One Gaussian is predicted per valid pixel in the UV texture map.
- Each UV Gaussian is bound to the local coordinate frame of its corresponding mesh triangle.
- **Inverse texture mapping** explicitly maps screen-space features to UV space.
- A mesh rasterizer filters out non-visible regions.
- The UV decoder first uses a StyleUNet to inpaint invisible regions, then applies a convolutional network to predict Gaussian attributes.

**Neural Refiner**: The total number of effective Gaussians may be fewer than 150K. A latent feature is attached to each Gaussian, splatted into a coarse feature map, and decoded by a StyleUNet refiner into the final high-quality image.

### Loss & Training

The total loss comprises an image loss, a position regularization term, and a scale regularization term. The image loss combines L1 and LPIPS over the full image, face crop, and hand crop, providing additional supervision on local details. Position regularization constrains UV Gaussians from straying too far from their parent triangles, and scale regularization prevents Gaussians from growing excessively large.

## Key Experimental Results

### Main Results

**Self-reenactment vs. 2D methods**:

| Method | PSNR | L1 | SSIM | LPIPS | FPS |
|---|---|---|---|---|---|
| MagicPose | 21.25 | 0.0333 | 0.8661 | 0.0913 | 0.12 |
| Champ | 22.01 | 0.0258 | 0.8643 | 0.1000 | 0.53 |
| MimicMotion | 24.46 | 0.0200 | 0.8768 | 0.0879 | 0.21 |
| **GUAVA** | **25.87** | **0.0162** | **0.9000** | **0.0813** | **52.21** |

**Self-reenactment vs. 3D methods**:

| Method | PSNR | SSIM | LPIPS | Input | Reconstruction Time |
|---|---|---|---|---|---|
| ExAvatar | 24.09 | 0.8783 | 0.1064 | Half-sequence video | ~2.4 h |
| GaussianAvatar | 23.62 | 0.8780 | 0.1085 | Half-sequence video | ~1.3 h |
| GART | 24.46 | 0.8805 | 0.1016 | Half-sequence video | ~7 min |
| **GUAVA** | **25.70** | **0.8976** | **0.0836** | Single frame | **~98 ms** |

**Cross-identity driving IPS (ArcFace identity preservation)**:

| GUAVA | MimicMotion | Champ | MagicPose |
|---|---|---|---|
| **0.5554** | 0.1310 | 0.3677 | 0.3277 |

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Full model | 25.87 | 0.9000 | 0.0813 |
| w/o refiner | 24.93 | 0.8851 | 0.1060 |
| w/o inverse texture mapping | 25.65 | 0.8977 | 0.0864 |
| w/o UV Gaussians | 25.82 | 0.8971 | 0.0877 |
| w/o EHM | 25.60 | 0.8950 | 0.0846 |

The refiner contributes most significantly (PSNR: 25.87 → 24.93), and EHM makes a notable contribution to facial expression accuracy.

### Key Findings

1. GUAVA is 100–400× faster than 2D methods (52 FPS vs. 0.12–0.53 FPS) while achieving superior quality.
2. Using only a single frame, it outperforms 3D methods that require half-sequence videos for training, reducing reconstruction time from hours to 98 ms.
3. Cross-identity driving IPS reaches 0.5554, far exceeding MimicMotion's 0.1310, demonstrating the identity consistency advantage of 3D representations.
4. The EHM model enables more accurate facial expression (blinking, talking) and hand gesture driving.
5. The training dataset comprises over 26,000 video clips and 620,000 frames covering diverse upper-body scenarios.

## Highlights & Insights

- **The dual-branch Gaussian design is elegant**: template Gaussians handle coarse geometry while UV Gaussians capture high-frequency details, forming a strong complementary pair.
- **Inverse texture mapping** explicitly projects 2D screen-space features into UV space, avoiding the difficulty of implicitly learning correspondences.
- **The EHM model** cleverly fuses SMPLX (body) and FLAME (face), leveraging the strengths of both.
- 98 ms reconstruction + 52 FPS rendering = a genuinely real-time-capable avatar system.
- Additional crop losses on face and hand regions are a simple yet effective technique for improving local detail.

## Limitations & Future Work

- Only the upper body is supported; full-body coverage (lower body, legs, feet) is not addressed.
- Reconstruction quality depends on EHM tracking accuracy; tracking failures under heavy occlusion degrade results.
- Training requires 156 GPU hours on A6000, which is non-trivial.
- Generalization to extreme poses or unseen clothing styles has not been thoroughly analyzed.
- The refiner introduces additional computation that may affect deployment in the most latency-sensitive scenarios.

## Related Work & Insights

- **GAGAvatar** achieves generalizable head Gaussian reconstruction; GUAVA extends this to the upper body.
- **ExAvatar** combines mesh and Gaussian representations; GUAVA's dual-branch design can be viewed as a generalizable counterpart.
- **MimicMotion/Champ** and similar 2D diffusion-based animation methods achieve high generation quality but poor identity consistency; GUAVA addresses this with a 3D representation.
- The inverse texture mapping idea is broadly applicable to other parametric model-based 3D reconstruction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First generalizable upper-body Gaussian avatar framework; EHM + dual-branch design is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparison against 2D and 3D methods; thorough ablations; multi-dimensional evaluation of speed, quality, and identity preservation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with rich illustrations.
- **Value**: ⭐⭐⭐⭐⭐ — Strong real-time usability; significant contribution to the avatar research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[ICCV 2025\] GAS: Generative Avatar Synthesis from a Single Image](gas_generative_avatar_synthesis_from_a_single_image.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads](facelift_learning_generalizable_single_image_3d_face_reconstruction_from_synthet.md)
- [\[ICCV 2025\] PHD: Personalized 3D Human Body Fitting with Point Diffusion](phd_personalized_3d_human_body_fitting_with_point_diffusion.md)

</div>

<!-- RELATED:END -->
