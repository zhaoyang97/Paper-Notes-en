---
title: >-
  [Paper Note] 3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations
description: >-
  [CVPR 2025][3D Vision][Head Avatar] A 3D Gaussian head avatar method with compact tensorial representation is proposed, which stores the static neutral-expression appearance using canonical tri-planes and the dynamic texture (opacity offset) of each blendshape using lightweight 1D feature lines. It achieves 300 FPS real-time rendering and accurate capture of dynamic facial details with only **10MB of storage**, comprehensively outperforming GA, GBS…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Head Avatar"
  - "Tensorial"
  - "Tri-plane"
  - "Feature Lines"
  - "Dynamic Texture"
  - "Compact"
date: 2026-05-08
content_hash: 61f4ad5a32b0561f
---

# 3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations

**Conference**: CVPR 2025  
**arXiv**: [2504.14967](https://arxiv.org/abs/2504.14967)  
**Code**: Not released  
**Area**: 3D Vision / Head Avatar Reconstruction / 3D Gaussian Splatting  
**Keywords**: Head Avatar, Tensorial, Tri-plane, Feature Lines, Dynamic Texture, Compact  

## TL;DR
A 3D Gaussian head avatar method with compact tensorial representation is proposed, which stores the static neutral-expression appearance using canonical tri-planes and the dynamic texture (opacity offset) of each blendshape using lightweight 1D feature lines. It achieves 300 FPS real-time rendering and accurate capture of dynamic facial details with only **10MB of storage**, comprehensively outperforming GA, GBS, and GHA in PSNR and storage efficiency on the Nersemble dataset.

## Background & Motivation
High-quality animatable head avatars must concurrently satisfy three requirements: (1) accurately capturing dynamic textures (such as wrinkles) that change with expression; (2) real-time rendering; (3) lightweight storage for easy transmission and deployment. Existing methods can only satisfy one or two of these: GA does not model dynamic textures; GBS stores full Gaussian sets for each blendshape, resulting in a storage size of 2GB; GHA captures details with an MLP and super-resolution but only runs at 20 FPS. No method has yet been able to achieve all three simultaneously.

## Core Problem
How to accurately model facial-expression-driven dynamic texture changes while maintaining real-time rendering and extremely low storage space?

## Method

### Overall Architecture
A three-level decoupled representation:
1. **Geometric Motion → FLAME Mesh**: Gaussians are bound to FLAME triangles and move with the mesh.
2. **Neutral Static Appearance → Canonical Tri-plane**: Replaces SH coefficients, storing view-dependent color.
3. **Dynamic Texture → 1D Feature Lines Blendshapes**: A set of 1D feature lines for each blendshape, decoded into opacity offset $\Delta\alpha$.

### Key Designs
1. **Canonical Tri-plane Color Representation**: Three orthogonal feature planes $T_{xy}, T_{xz}, T_{yz}$ (128×128) replace the 48-parameter SH coefficients in 3DGS. For any canonical position, bilinear interpolation + concatenation are performed, followed by decoding into RGB colors through a tiny MLP. The front plane $T_{xy}$ uses a larger feature dimension (32), while the side planes $T_{xz}, T_{yz}$ use smaller dimensions (16), leveraging the prior that facial information is concentrated on the front face.

2. **1D Feature Lines Dynamic Texture**: FLAME has 100 PCA blendshapes, which would be too costly to store using 2D planes for each. It innovatively uses 1D feature lines (spatial resolution 64, feature dimension 32) projected along the x/y/z axes and concatenated, significantly compressing the storage. The expression coefficients perform linear interpolation on the feature lines to obtain the feature line of the current frame, which is decoded into opacity offset $\Delta\alpha$ by an MLP. Storage: each group is only ~25KB, requiring only 2.41MB for 80 blendshapes + 16 jaw bases.

3. **Adaptive Truncated Opacity Penalty**: Calculates the displacement $|\bar{t}|$ of mesh triangles relative to the neutral expression for each frame, and the opacity offset in the static area (displacement < threshold $\tau$) is constrained to approach 0. This decouples static appearance and dynamic details, improving generalization to unseen expressions.

4. **Class-Balanced Sampling**: There are few frames of large expressions in the training data. Spectral clustering (16 classes) is performed based on the similarity of FLAME mesh vertex displacements, and all classes are uniformly sampled during training to avoid overfitting to small dynamic frames. The weights of the eye vertices are doubled.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{image} + \mathcal{L}_{geom} + \mathcal{L}_{op}$$
- $\mathcal{L}_{image} = 0.8\mathcal{L}_1 + 0.2\mathcal{L}_{D-SSIM}$
- $\mathcal{L}_{geom}$: Position loss + scale loss (constraining Gaussians to be close to the mesh)
- $\mathcal{L}_{op}$: Adaptive truncated opacity offset penalty
- Adam optimizer, 600K iterations/subject, PyTorch

## Key Experimental Results

### Nersemble Dataset (9 subjects, 16 views)

#### Novel View Synthesis

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Storage↓ | FPS↑ |
|------|-------|-------|--------|-------|------|
| GA | 31.47 | 0.949 | 0.051 | 21MB | 330 |
| GHA | 26.99 | 0.935 | 0.049 | 120MB | 20 |
| GBS | 28.90 | 0.950 | 0.063 | **2GB** | 370 |
| **Ours** | **32.97** | **0.951** | 0.059 | **10MB** | 300 |

#### Self-Reenactment (Generalization to Unseen Expressions)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| GA | 27.27 | 0.923 | 0.067 |
| GBS | 25.98 | 0.927 | 0.081 |
| **Ours** | **28.07** | **0.926** | 0.077 |

### Ablation Study (subject #306)
- **Tri-planes**: Avoids overfitting, improving generalization to unseen expressions (PSNR +0.01 in NVS, but +0.1+ in self-reenactment).
- **Feature Lines**: Models dynamic texture, resulting in NVS PSNR +1.95 and self-reenactment +0.1.
- **Opacity Offset vs. Position/Rotation/Scale Offsets**: Tuning opacity only yields better generalization (self-reenactment 31.64 vs. 30.97), as more degrees of freedom lead to overfitting.
- **Balanced Sampling**: Self-reenactment PSNR increased by ~0.2.
- **Truncated Penalty**: Prevents floaters around the hair and artifacts in the teeth area.

## Highlights & Insights
- **Extreme Storage Efficiency**: 10MB/subject (tri-plane 4.05MB + feature lines 2.41MB + geometry 3.25MB), 200x smaller than GBS (2GB).
- **1D Feature Lines Modeling Dynamic Texture**: Clever dimensionality reduction—compressing 2D/3D tensors to 1D, exploiting the low-rank characteristics of facial dynamics.
- **Opacity Offset Design**: Adjusting only opacity instead of position/rotation/scale reduces the degrees of freedom and enhances generalization.
- **Asymmetric Tri-plane Dimensions**: Optimizes storage by leveraging the prior that facial information is concentrated on the front face.

## Limitations & Future Work
- Relies on the tracking quality of FLAME mesh, failing to handle complex hairstyles or topological changes inside the oral cavity; the expressive space of FLAME itself is limited (100 PCA bases), and extreme expressions may fall outside the modeling range.
- Does not decouple material and lighting, making it impossible to perform avatar relighting, which limits applicability in changing illumination scenes.
- Dynamic texture only models changes in opacity; dynamic color variations (e.g., blushing, skin tone changes) are not explicitly modeled, limiting the ability to capture subtle expressions.
- Evaluated primarily on the Nersemble dataset (multi-view controlled environment), untested on other datasets (such as monocular videos outside NeRSemble), and the generalization ability under monocular scenarios remains unknown.
- Training requires 600K iterations/subject, making the per-subject training cost still high; lacks priors for cross-identity generalization, requiring retraining for new users.
- The tri-plane resolution is fixed at 128×128, which may be insufficient for high-frequency details (such as wrinkles around the eyes).

## Related Work & Insights
- **GA (Gaussian Avatars)**: Does not model dynamic texture, yielding 1.5 lower PSNR. The proposed method adds dynamic details while maintaining real-time performance.
- **GBS (Gaussian Blendshapes)**: Stores full Gaussians for each blendshape, resulting in 2GB storage, which is 200x larger than this method, and introduces artifacts when linearly interpolating large expressions.
- **GHA (Gaussian Head Avatar)**: Employs MLP + super-resolution, rendering with the lowest PSNR (26.99) and only at 20 FPS. The proposed method achieves 300 FPS and a PSNR that is 6 points higher.
- **INSTA**: NeRF-based method, rendering speed is far lower than 3DGS solutions (<5 FPS), with large storage overhead; this method also yields significantly higher PSNR in NVS.
- **FlashAvatar**: Also uses tri-planes + 3DGS, but lacks dynamic texture modeling, and symmetric tri-plane dimension allocation fails to utilize the prior of frontal-focused facial information.
- **PointAvatar / DECA**: Point-cloud or decoupled-encoding methods, inferior to the proposed 3DGS scheme in rendering resolution and real-time performance.

## Related Work & Insights
- The compression concept of 1D feature lines can be extended to other tasks requiring compact representation of dynamic scenes (such as 4D scene reconstruction, hand gesture modeling, and full-body avatars).
- The generalization discovery of "opacity-only offset" holds reference value for other avatar methods—limiting degrees of freedom actually enhances generalization, which is consistent with regularization theory.
- The design of asymmetric tri-plane dimensions inspires: adjusting representational capacity based on prior knowledge of information distribution rather than uniform allocation.
- The concept of adaptive truncated penalty—guiding appearance learning with geometric signals (mesh displacement)—can be transferred to other joint geometry-appearance modeling tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ 1D feature lines replace 2D/3D tensors as the core innovation, with insightful opacity-only offset design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three tasks + extensive ablations + extreme view/expression testing + comparison with INSTA.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, fair comparison.
- Value: ⭐⭐⭐⭐ 10MB + 300FPS avatars have direct practical value for mobile applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Steepest Descent Density Control for Compact 3D Gaussian Splatting](steepest_descent_density_control_for_compact_3d_gaussian_splatting.md)
- [\[CVPR 2025\] GASP: Gaussian Avatars with Synthetic Priors](gasp_gaussian_avatars_with_synthetic_priors.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](../../CVPR2026/3d_vision/physhead_simulation-ready_gaussian_head_avatars.md)
- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](../../ICCV2025/3d_vision/strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [\[ICLR 2026\] FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](../../ICLR2026/3d_vision/fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)

</div>

<!-- RELATED:END -->
