---
title: >-
  [Paper Note] NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction
description: >-
  [CVPR2025][Human Understanding][head avatar] NBAvatar proposes the Neural Billboard primitive, which combines learnable planar geometric primitives with deferred neural texture rendering to achieve photorealistic head avatar rendering under hand-face interaction, reducing LPIPS by 30% compared to Gaussian-based methods at megapixel resolution.
tags:
  - "CVPR2025"
  - "Human Understanding"
  - "head avatar"
  - "hand-face interaction"
  - "neural rendering"
  - "billboard splatting"
  - "deferred neural rendering"
date: 2026-05-08
content_hash: be9acca8ea4386c0
---

# NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction

**Conference**: CVPR2025  
**arXiv**: [2603.12063](https://arxiv.org/abs/2603.12063)  
**Code**: [Project Page](https://david-svitov.github.io/NBAvatar_project_page)  
**Area**: Human Understanding  
**Keywords**: head avatar, hand-face interaction, neural rendering, billboard splatting, deferred neural rendering

## TL;DR
NBAvatar proposes the Neural Billboard primitive, which combines learnable planar geometric primitives with deferred neural texture rendering to achieve photorealistic head avatar rendering under hand-face interaction, reducing LPIPS by 30% compared to Gaussian-based methods at megapixel resolution.

## Background & Motivation
1. Hand-face interaction is a critical information source in human communication, which is essential for telepresence and VR applications.
2. Modern methods focus on rendering head or hands separately, neglecting non-rigid deformations and color variations caused by hand-face interactions.
3. Although 3DGS methods offer high quality, they suffer from inherent artifacts: blurry facial textures and prominent Gaussians at body boundaries.
4. InteractAvatar is the state-of-the-art hand-face interaction method, which uses an MLP to predict spatial offsets in interaction regions, but it inherits 3DGS artifacts.
5. DNR (Deferred Neural Rendering) achieves real-time speed and high fidelity, but its original design is based on fixed mesh parameterization.
6. Adapting the neural texture paradigm to independently transformable planar primitives presents a non-trivial optimization challenge.

## Method

### Overall Architecture
NBAvatar consists of three stages: (1) fitting FLAME/MANO parametric models from multi-view videos and using PBD physical simulation for coarse facial deformation; (2) anchoring Neural Billboard primitives to the mesh surface polygons; (3) rendering neural feature maps and then generating the final RGB image via a UNet decoder.

### Key Designs

**1. Neural Billboard Primitives**
- Parameterization: $\{\mu_i, s_i, r_i, T_i^{NT}, T_i^{\alpha}\}$ (position, scale, rotation, 16×16 six-channel neural texture, single-channel alpha texture).
- Replaces the RGB texture of Billboard Splatting with a learnable neural feature map.
- Alpha textures are initialized from a Gaussian distribution to learn the visibility of each planar point.
- Accumulating neural texture values along camera rays: $c(x) = \sum_i T_i^{NT}[\mathbf{u}(x)] T_i^{\alpha}[\mathbf{u}(x)] \prod_{j=1}^{i-1}(1-T_j^{\alpha}[\mathbf{u}(x)])$
- Produces a 6-channel feature map $I_f^{NB}$ and a 1-channel alpha map $I_\alpha^{NB}$.

**2. UNet Deferred Renderer**
- Decodes the 6-channel rasterized feature map into an RGB image and a transparency map.
- Provides high-frequency details and an interaction-aware inductive bias.
- Hand and face features are rasterized in a shared screen space, where neighboring features naturally modulate the decoder response.
- **Without using an explicit interaction conditioning module**, it implicitly captures contact dynamics relying entirely on spatial feature aggregation.

**3. Decoupled Geometry and Appearance Training**
- Key challenge: spatial drift of the billboards and neural features compete to explain contour and shading variations.
- Introduces **intermediate silhouette supervision** during the billboard rasterization stage: $\mathcal{L}_{NB} = \lambda_{NB} \text{Dice}(I_\alpha^{NB}, GT_\alpha)$.
- Ensures that billboards closely align with the GT contours, decoupling rigid geometry from view-/pose-dependent appearance.
- KNN regularization constrains rotation/scale consistency of adjacent billboards + position offset regularization.

**4. Avatar Animation Control**
- Billboards are anchored to mesh polygons and driven by polygon transformations $\{T_i, R_i, k_i\}$.
- Position: $\mu_i' = k_i R_i \mu_i + T_i$, scale: $s_i' = k_i s_i$, rotation: $q_i' = R_i q_i$.

### Loss & Training
- RGB Training: MSE + $\lambda_{lpips}$ LPIPS (full image for the first 40K iterations, then 256×256 random crops afterwards).
- Silhouette Supervision: $\lambda_{NB} = 0.1$ Dice loss.
- Regularization: KNN + position offset $\lambda_\Delta = 0.001$.

## Key Experimental Results

### Novel-View Synthesis (Decaf Dataset, 1024×1024)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SplattingAvatar | 25.17 | 0.955 | 0.080 |
| GaussianAvatars | 25.31 | 0.957 | 0.076 |
| **NBAvatar** | **25.65** | **0.958** | **0.056** |

LPIPS is reduced by 26.3% (vs GA) and 30.0% (vs SA).

### Self-Reenactment (Held-out Pose)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SplattingAvatar | 25.82 | 0.962 | 0.066 |
| GaussianAvatars | 25.04 | 0.960 | 0.066 |
| **NBAvatar** | **25.48** | **0.961** | **0.052** |

### Comparison with InteractAvatar (512×512, IA Evaluation Protocol)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| InteractAvatar | 29.85 | 0.933 | 0.034 |
| NBAvatar | 24.41 | **0.936** | 0.051 |

SSIM is higher, but PSNR is lower than IA (due to differences in preprocessing). Qualitative comparisons show sharper facial details for NBAvatar.

### Ablation Study (Subject 2 Novel-View)

| Configuration | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| w/o $\mathcal{L}_{NB}$ | 26.75 | 0.9711 | 0.039 |
| w/o $\mathcal{L}_{Reg}$ | 28.43 | 0.9748 | 0.033 |
| w/o DNR | 25.88 | 0.9655 | 0.045 |
| **Full** | **28.63** | **0.976** | **0.032** |

## Highlights & Insights
1. **Neural Billboard Primitives**: An elegant combination of explicit geometry and implicit neural features, offering both the surface alignment of billboards and the expressiveness of neural textures.
2. **Implicit Interaction Modeling**: Instead of relying on explicit interaction modules, it leverages the spatial receptive field of UNet to implicitly capture hand-face contact dynamics, making it simpler and more generalizable.
3. **Decoupled Silhouette Supervision**: The intermediate Dice loss is key to stable joint optimization, and the ablation study confirms its indispensability for reducing artifacts.
4. **Megapixel Quality**: At 1024×1024 resolution, it significantly reduces typical boundary and blur artifacts common in 3DGS.
5. **Cross-Subject Reenactment**: Supports hand-face pose transfer across different actors.

## Limitations & Future Work
1. The UNet renderer increases inference overhead, and real-time performance is not reported (presumably lower than pure 3DGS-based methods).
2. It relies heavily on the 3DMM fitting quality of FLAME/MANO, and fitting errors directly degrade the rendering quality.
3. Validated only on the Decaf dataset (4 subjects); the generalizability requires further testing.
4. Hand representation remains relatively coarse, and details of highly articulated fingers may be insufficient.

## Related Work & Insights
- Neural Billboard is a natural fusion of DNR and Billboard Splatting, demonstrating the potential of explicit-implicit hybrid representations.
- The concept of implicit interaction modeling can be generalized to other multi-body interaction rendering scenarios (e.g., hand-object or human-human interactions).
- The silhouette supervision training strategy can be applied to other tasks requiring geometry-appearance decoupling.
- It establishes a new baseline for hand-face interaction rendering in telepresence and social VR.

## Rating
- Novelty: ⭐⭐⭐⭐ (Neural Billboard primitives + implicit interaction modeling)
- Experimental Thoroughness: ⭐⭐⭐ (Tested only on 4 subjects within the Decaf dataset)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation and ablation studies)
- Value: ⭐⭐⭐⭐ (Advances the quality of hand-face interaction rendering)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](../../ICCV2025/human_understanding/dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)
- [\[CVPR 2025\] VI3NR: Variance Informed Initialization for Implicit Neural Representations](vi3nr_variance_informed_initialization_for_implicit_neural_representations.md)
- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[CVPR 2025\] RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars](rgbavatar_reduced_gaussian_blendshapes_for_online_modeling_of_head_avatars.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)

</div>

<!-- RELATED:END -->
