---
title: >-
  [Paper Note] HRAvatar: High-Quality and Relightable Gaussian Head Avatar
description: >-
  [CVPR 2025][3D Vision][Head Reconstruction] HRAvatar proposes a monocular video head reconstruction method based on 3DGS, which achieves flexible deformation through learnable blendshapes and LBS, reduces tracking errors using an end-to-end expression encoder, and introduces a physically-based rendering model to achieve high-quality real-time relighting.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Head Reconstruction"
  - "3D Gaussian Splatting"
  - "Relighting"
  - "Animatable"
  - "Monocular Video"
date: 2026-05-08
content_hash: 2d438cfb0c40c4e2
---

# HRAvatar: High-Quality and Relightable Gaussian Head Avatar

**Conference**: CVPR 2025  
**arXiv**: [2503.08224](https://arxiv.org/abs/2503.08224)  
**Code**: Yes (provided on project page)  
**Area**: 3D Vision  
**Keywords**: Head Reconstruction, 3D Gaussian Splatting, Relighting, Animatable, Monocular Video  

## TL;DR
HRAvatar proposes a monocular video head reconstruction method based on 3DGS, which achieves flexible deformation through learnable blendshapes and LBS, reduces tracking errors using an end-to-end expression encoder, and introduces a physically-based rendering model to achieve high-quality real-time relighting.

## Background & Motivation

**Background**: Reconstructing animatable 3D head avatars from monocular videos is a critical requirement in fields such as film, gaming, and AR/VR. Recently, 3DGS-based methods (e.g., Splatting-avatar, Flash-avatar) have achieved real-time rendering by binding Gaussian points onto parametric head models (e.g., FLAME), but their reconstruction quality remains limited.

**Limitations of Prior Work**: Existing methods have three core limitations: (1) Insufficient deformation flexibility—rigidly binding Gaussian points to the mesh polygons of generic parametric models fails to capture personalized facial deformation details; (2) Inaccurate expression tracking—pre-tracked FLAME parameters obtained by fitting pseudo-2D keypoints introduce propagation errors that degrade reconstruction quality; (3) Inability to relight—directly fitting colors couples the subject's intrinsic appearance with environmental lighting.

**Key Challenge**: The contradiction between the generality of parametric head models and the demand for personalized deformations, as well as the ambiguity of appearance decomposition under unknown illumination.

**Goal**: (1) Make Gaussian point deformations more flexible and personalized; (2) Reduce the impact of expression tracking errors on reconstruction quality; (3) Achieve realistic relighting under unknown illumination from monocular videos.

**Key Insight**: The authors observe that each person's facial shape and deformation patterns are unique, which generic parametric models fail to express accurately. Therefore, they independently learn blendshape basis functions and blend weights for each Gaussian point, and utilize an end-to-end trained expression encoder to replace the independent pre-tracking step.

**Core Idea**: Replace rigid FLAME binding with learnable point-wise blendshapes and LBS, combined with an end-to-end expression encoder and a physically-based rendering model, to achieve high-quality, relightable head avatar reconstruction.

## Method

### Overall Architecture
The input is a monocular head video of $M$ frames under unknown illumination, and the output is an animatable, relightable 3D head avatar. The overall pipeline is divided into three stages: (1) First, iteratively optimize and fix shape parameters $\beta$ and pose parameters $\{\theta_j\}$ through pre-tracking; (2) Estimate expression parameters $\psi$ and jaw pose $\theta^{jaw}$ in an end-to-end manner via an expression encoder; (3) Transform Gaussian points from the canonical space to the pose space using learnable linear blendshapes and LBS, subsequently render albedo, roughness, reflectance, and normal maps, and finally compute pixel colors via physically-based rendering (PBR).

### Key Designs

1. **End-to-End Accurate Expression Tracking (Expression Encoder)**:

    - **Function**: Accurately estimate facial expression parameters, reducing the impact of pre-tracking errors on reconstruction quality.
    - **Mechanism**: Use a pre-trained SMIRK encoder $\mathcal{E}$ to input the current frame image $I$ and output the expression parameters $\psi$ and jaw pose $\theta^{jaw}$. Crucially, this encoder is optimized end-to-end during training via photometric loss, rather than relying on pre-tracked results from pseudo-2D keypoints. Additionally, a jaw pose regularization loss $\mathcal{L}_{jaw}$ is introduced to constrain the distance between the inferred and pre-tracked values, preventing excessive drift.
    - **Design Motivation**: Traditional fitting methods optimize parameters using pseudo-labels, which leads to large errors, whereas end-to-end training leverages real-image supervision to both improve precision and maintain generalization. Prior methods like PointAvatar directly optimize parameters, which introduces train/test inconsistencies.

2. **Learnable Linear Blendshapes and LBS (Learnable Deformation)**:

    - **Function**: Achieve flexible personalized geometric deformation, mapping Gaussian points from the canonical space to the pose space.
    - **Mechanism**: Introduce three sets of learnable blendshape bases for each Gaussian point: shape base $S$, expression base $E$, and pose base $P$, together with learnable blend weights $\mathcal{W}$. The deformation process occurs in two steps: first, linear blendshapes compute shape/expression/pose offsets ($X_e = X_c + \mathcal{BS}(\psi, E) + \mathcal{BS}(\mathcal{R}(\theta^*) - \mathcal{R}(\theta^0), P)$); second, joint-driven rigid transformations are applied via LBS ($X_p = R_{lbs}X_e + T_{lbs}$). Initialization leverages linear interpolation from FLAME mesh faces to provide priors.
    - **Design Motivation**: Unlike methods such as GBS that use a shared MLP, learning basis functions and weights independently per point better captures personalized deformations, especially in non-standard regions like hair and accessories. Experiments demonstrate that this strategy outperforms the shared MLP scheme.

3. **Physically-Based Shading (Physically-Based Shading)**:

    - **Function**: Decompose facial appearance into multiple physical attributes to enable realistic relighting effects.
    - **Mechanism**: Define three attributes for each Gaussian point: albedo $a$, roughness $o$, and Fresnel base reflectance $f_0$. During rendering, albedo, roughness, reflectance, and normal maps are first obtained via rasterization, and then the specular reflection $I_{specular}$ and diffuse reflection $I_{diffuse} = \mathbf{A} \cdot I_{irr}(\mathbf{N})$ are computed using Split-Sum approximation and a BRDF model. During training, two cube maps (an environmental irradiance map $I_{irr}$ and a pre-filtered environment map $I_{env}$) are optimized. Additionally, an albedo pseudo-prior loss $\mathcal{L}_{albedo}$ and a normal consistency loss $\mathcal{L}_{normal}$ are introduced to ensure physically plausible material decomposition.
    - **Design Motivation**: Directly fitting color using spherical harmonics (SH) representation cannot support relighting. Although a physically-based rendering model offers slightly less parametric flexibility, it maintains comparable reconstruction quality while supporting real-time relighting and material editing. The albedo prior prevents local lighting effects from being wrongly coupled into the albedo.

### Loss & Training
The total loss is: $\mathcal{L}_{total} = \mathcal{L}_{rgb} + 0.1\mathcal{L}_{jaw} + 10^{-5}\mathcal{L}_{normal} + 0.25\mathcal{L}_{albedo} + 0.02\mathcal{L}_{tv}(\mathbf{O})$. Here, $\mathcal{L}_{rgb}$ combines MAE (weight 0.8) and D-SSIM (weight 0.2), and $\mathcal{L}_{tv}$ is the total variation loss for the roughness map to guarantee smoothness. The original point densification and pruning strategies of 3DGS are preserved.

## Key Experimental Results

### Main Results

| Dataset | Metric | HRAvatar | GBS | Flash-avatar | Gain |
|--------|------|----------|-----|-------------|------|
| INSTA (10 subjects) | PSNR↑ | **30.36** | 29.64 | 29.13 | +0.72 |
| INSTA | LPIPS↓ | **0.0569** | 0.0823 | 0.0719 | -30.9% |
| HDTF (8 subjects) | PSNR↑ | **28.55** | 27.81 | 27.58 | +0.74 |
| HDTF | LPIPS↓ | **0.0825** | 0.1297 | 0.1095 | -36.4% |
| Self-captured (5 subjects) | PSNR↑ | **28.97** | 28.59 | 27.46 | +0.38 |
| Self-captured | LPIPS↓ | **0.1059** | 0.1560 | 0.1456 | -32.1% |

Rendering speed is approximately **155 FPS**, supporting real-time animation and relighting.

### Ablation Study

| Configuration | PSNR↑ | MAE*↓ | SSIM↑ | LPIPS↓ |
|------|-------|-------|-------|--------|
| Full model | **30.36** | **0.845** | **0.9482** | **0.0569** |
| Rigged to FLAME | 29.79 | 0.937 | 0.9431 | 0.0695 |
| MLP deform | 29.67 | 0.966 | 0.941 | 0.0706 |
| w/o exp. encoder | 29.70 | 0.933 | 0.9438 | 0.0667 |
| w/o learnable deform | 29.83 | 0.923 | 0.9440 | 0.0684 |
| w/o PBS | 30.34 | 0.850 | 0.9480 | 0.0563 |

### Key Findings
- The learnable deformation model contributes the most: "rigged to FLAME" drops by 0.57 PSNR, and "MLP deform" drops by 0.69 PSNR, verifying that the point-wise independent learning strategy is optimal.
- The expression encoder is highly effective: removing it drops PSNR by 0.66, and fine expressions such as mouth shapes and blinking visually degrade significantly.
- The PBS mode performs almost on par with standard 3DGS on reconstruction metrics (only a 0.02 PSNR drop) while successfully gaining relighting capability.
- Without $\mathcal{L}_{albedo}$, albedo and specular highlights become coupled, resulting in unrealistic relighting; without $\mathcal{L}_{normal}$, the normal map becomes chaotic, causing blocky artifacts in relighting.

## Highlights & Insights
- **Point-wise independent learning of blendshapes**: Unlike the global Gaussian basis in GBS or the shared MLP in PointAvatar, this method learns deformation basis functions independently for each Gaussian point, achieving the most flexible personalized deformation modeling. This "space-for-accuracy" design concept can be transferred to other tasks like full-body reconstruction.
- **Ingenious co-optimization of expression tracking and reconstruction**: The encoder is trained end-to-end but with jaw regularization, striking an excellent balance between accuracy improvement and training stability. This approach is highly applicable to any scenarios requiring joint parameter estimation and downstream task optimization.
- **No performance drop with PBS approximation**: The physically-based rendering branch achieves relighting capabilities while maintaining almost the same PSNR as appearance-fitting-only methods, demonstrating that approximate physical models are sufficiently effective for head scenes.

## Limitations & Future Work
- When training data is insufficient, the model is still constrained by the FLAME prior, leading to limited control over non-standard elements like hair and accessories.
- Some shadows or wrinkles might be incorrectly coupled into the albedo or reflectance, causing flaws in specularity and shadow relighting results.
- Unable to handle full-head reconstruction under unknown camera poses—facial keypoints become unreliable when the yaw angle approaches 90 degrees.
- Future Directions: Integrate better albedo estimation models (e.g., diffusion priors), extend to full-body reconstruction, and support more complex lighting/rendering models (e.g., subsurface scattering).

## Related Work & Insights
- **vs. Flash-avatar**: Flash-avatar rigidly binds Gaussian points to the FLAME mesh, which limits deformation flexibility; ours learns point-wise deformation independently, achieving a $+1.2$ PSNR gain and a major improvement in LPIPS.
- **vs. GBS (3D Gaussian Blendshapes)**: GBS learns a global Gaussian basis to handle expressions but performs poorly on pose variations; ours handles both expressions and pose variations while introducing LBS and relighting.
- **vs. FLARE**: FLARE performs relighting using a mesh and BRDF but suffers from limited reconstruction quality and noisy normals; ours utilizes 3DGS to obtain higher quality and smoother normals.
- **vs. PointAvatar**: PointAvatar uses a shared MLP to predict deformations and directly optimizes parameters, requiring post-optimization during testing; ours utilizes an encoder to preserve generalization capability.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of point-wise learnable blendshapes + LBS + end-to-end expression encoder + physical rendering is highly complete, though individual techniques are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 subjects across three datasets, comprehensive ablation studies, comparison with 5 baselines, including relighting and cross-reenactment experiments.
- Writing Quality: ⭐⭐⭐⭐ Well-organized with well-explained motivations, though some mathematical notations are quite dense.
- Value: ⭐⭐⭐⭐ Achieves high-quality, real-time, and relightable monocular head reconstruction, showing high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RNG: Relightable Neural Gaussians](rng_relightable_neural_gaussians.md)
- [\[CVPR 2025\] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] Synthetic Prior for Few-Shot Drivable Head Avatar Inversion](synthetic_prior_for_few-shot_drivable_head_avatar_inversion.md)
- [\[CVPR 2026\] RelightAnyone: A Generalized Relightable 3D Gaussian Head Model](../../CVPR2026/3d_vision/relightanyone_a_generalized_relightable_3d_gaussian_head_model.md)

</div>

<!-- RELATED:END -->
