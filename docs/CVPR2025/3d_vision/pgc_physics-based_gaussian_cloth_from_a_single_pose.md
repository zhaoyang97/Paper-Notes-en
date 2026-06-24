---
title: >-
  [Paper Note] PGC: Physics-Based Gaussian Cloth from a Single Pose
description: >-
  [CVPR 2025][3D Vision][3D Gaussians] This paper proposes PGC, a method to reconstruct simulatable, realistic garment assets from only a single-frame multi-view capture. By incorporating a hybrid strategy of mesh-embedded 3D Gaussians and physically-based rendering (PBR), the method achieves garment rendering in novel poses with both high-frequency details and correct lighting effects.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussians"
  - "Cloth Simulation"
  - "Physically-Based Rendering"
  - "Hybrid Representation"
  - "Single-frame Reconstruction"
date: 2026-05-08
content_hash: e08ec506e4ad29ee
---

# PGC: Physics-Based Gaussian Cloth from a Single Pose

**Conference**: CVPR 2025  
**arXiv**: [2503.20779](https://arxiv.org/abs/2503.20779)  
**Code**: None (Project Page: phys-gaussian-cloth.github.io)  
**Area**: 3D Vision / Garment Reconstruction  
**Keywords**: 3D Gaussians, Cloth Simulation, Physically-Based Rendering, Hybrid Representation, Single-frame Reconstruction

## TL;DR

This paper proposes PGC, a method to reconstruct simulatable, realistic garment assets from only a single-frame multi-view capture. By incorporating a hybrid strategy of mesh-embedded 3D Gaussians and physically-based rendering (PBR), the method achieves garment rendering in novel poses with both high-frequency details and correct lighting effects.

## Background & Motivation

**Background**: Garment reconstruction and animation are key technologies for virtual humans. Traditional mesh-based methods are naturally suited for physical simulation but struggle to represent high-frequency geometric details (e.g., fuzz, pockets, zippers). Recently, 3D Gaussian Splatting (3DGS) has shown excellent performance in appearance reconstruction, capturing fine volumetric details, but Gaussians lack driving and simulation capabilities.

**Limitations of Prior Work**: (1) Mesh-based methods are limited by mesh resolution, failing to represent high-frequency surface details like fuzz or knitting; (2) Pure 3DGS methods "bake" the lighting information (e.g., shadows, specular highlights) of the training frames, leading to incorrect lighting effects in novel poses; (3) Applying 3DGS to garments typically requires multi-frame tracking (e.g., Gaussian Garments requires video sequence tracking), where the tracking itself is computationally expensive and inaccuracies lead to blurriness.

**Key Challenge**: An inherent contradiction exists between high-quality garment appearance (requiring volumetric details of 3DGS) and generalization to novel poses (requiring correct lighting response and physical simulation capabilities). Moreover, existing methods either require large amounts of multi-frame data or sacrifice appearance quality.

**Goal**: To reconstruct a garment asset with both fine appearance and physical simulatability from only a single-frame multi-view capture.

**Key Insight**: It is observed that the image signal can be decomposed into low-frequency (pose-dependent lighting, shadows, and other far-field effects) and high-frequency (fabric textures, fuzz, and other near-field details) components. The low-frequency part can be correctly computed under novel poses using traditional PBR methods, while the high-frequency part remains largely invariant under pose changes and can be captured statically by 3DGS.

**Core Idea**: To construct a hybrid representation of mesh-embedded 3DGS + PBR. During rendering, high-pass filtering is used to extract high-frequency details from 3DGS, and low-pass filtering is used to extract low-frequency lighting from PBR rendering. Combining both yields the final image with both details and correct lighting.

## Method

### Overall Architecture

The input is a multi-view capture under a static pose (using a hemisphere of 170 cameras), and the output is a simulatable garment asset. The pipeline consists of three core components: (1) Mesh-Embedded 3DGS Reconstruction—anchoring Gaussians onto the reconstructed mesh and optimizing their appearance; (2) PBR Appearance Reconstruction—estimating the albedo map and fabric-specific BRDF reflectance parameters; (3) Hybrid Rendering—after driving mesh deformation with a physics simulator under novel poses, combining the high-frequency information from 3DGS and the low-frequency information from PBR to obtain the final rendering.

### Key Designs

1. **Mesh-Embedded 3DGS Reconstruction**:

    - **Function**: Reconstructing a Gaussian representation anchored to the mesh from multi-view captures to capture fine fabric details.
    - **Mechanism**: Garment meshes are first obtained from multi-view images via stereo matching and surface reconstruction, followed by remeshing to make them simulator-ready. One million Gaussian points are sampled on the mesh surface, where the position and rotation of each Gaussian are defined within the local coordinate system of its corresponding triangular face. Two losses are optimized: a reconstruction loss $\mathcal{L}_{3DGS} = \lambda \|I_k - G_k\|_1 + (1-\lambda) \text{SSIM}(I_k, G_k)$ and a foreground regularization loss $\mathcal{L}_{fg}$ (which aligns the cumulative opacity of the Gaussians with the foreground mask). Training is performed on full images (rather than segmented garment images) to prevent inaccurate segmentation from affecting fuzzy boundary details.
    - **Design Motivation**: Anchoring Gaussians to the mesh ensures that they faithfully follow the mesh deformation while maintaining the advantage of Gaussians in capturing volumetric details. Training on full images allows the model to learn fuzzy details on garment boundaries, such as fuzz and stray fibers.

2. **Fabric-Specific PBR Appearance Model**:

    - **Function**: Providing far-field shading capabilities that correctly respond to novel poses and lighting conditions for the garment.
    - **Mechanism**: The model is based on the Disney BRDF but replaces the sheen model. Observing that fabrics exhibit significant forward and backward scattering at grazing angles (due to stray surface fibers), which the built-in sheen component in the Disney BRDF fails to match, the authors adopt a sheen BRDF $f_s$ based on optical simulations that account for multiple fiber scattering. The final appearance model is $f = \sigma_s f_s + H_s(\mathbf{o}) \sigma_d(\mathbf{x}) f_d$, where $\sigma_d$ is the spatially-varying albedo, $\sigma_s$ is the sheen color, and $H_s$ is the sheen transmission term ensuring energy conservation. The albedo is back-projected into the texture space after separating illumination using a pre-trained intrinsic image decomposition network. The remaining parameters (roughness, sheen color, sheen roughness) are jointly optimized on the training frames via Mitsuba differentiable rendering.
    - **Design Motivation**: The Lambertian model entirely lacks sheen effects, while the standard Disney BRDF overestimates forward scattering. The fabric-specific sheen model more accurately matches the appearance of real fabrics, significantly improving PSNR.

3. **Gaussian-PBR Hybrid Rendering**:

    - **Function**: Combining high-frequency details and low-frequency lighting to achieve high-quality rendering under novel poses.
    - **Mechanism**: A physics simulator (XPBD) is used to generate the mesh vertex positions $V_t$ under the novel pose. For far-field shading, the PBR model is used to render the texture $T_t$ under the new pose in texture space, passing the color to the zero-order spherical harmonics $\phi_t^{\circ}$ of the Gaussians, which yields $S_t$ via the 3DGS renderer. For near-field shading, the originally optimized Gaussians are rendered under the novel pose with full spherical harmonics to obtain $G_t$, and an alpha-weighted Gaussian blur is applied for high-pass filtering $h(G_t) = G_t - l(G_t)$. The final composition is $\hat{I}_t = h(G_t) + l(S_t)$, where the high-pass term retains fabric texture details and the low-pass term provides correct wrinkle shadows and indirect illumination.
    - **Design Motivation**: Pose-dependent effects (shadows, indirect illumination) reside primarily in the low-frequency signal and can be correctly computed by PBR; high-frequency signals (fabric texture, fuzz) remain largely unchanged under pose changes and can be extracted from the original 3DGS. This frequency-domain decomposition strategy elegantly solves the "baked lighting" issue.

### Loss & Training

The 3DGS optimization employs standard L1+SSIM reconstruction loss and a foreground regularization loss. The PBR parameter optimization uses Mitsuba differentiable rendering. Simulation is conducted using the XPBD method on a SMPL-like parametric body. Garment meshes are automatically isolated using image segmentation. The entire pipeline does not require video sequence tracking, needing only a single-frame multi-view capture.

## Key Experimental Results

### Main Results

| Method | FSIM↑ | LPIPS (×10⁻²)↓ |
|------|-------|-----------------|
| SCARF | 0.764 | 5.00 |
| Animatable Gaussians | 0.827 | 3.39 |
| **PGC (Ours)** | **0.834** | **3.38** |

### Ablation Study

| Configuration | FSIM↑ | LPIPS (×10⁻²)↓ |
|------|-------|-----------------|
| 3DGS-Only | 0.825 | 3.41 |
| PBR-Only | 0.809 | 4.67 |
| **Full (Hybrid)** | **0.834** | **3.38** |

### Key Findings

- 3DGS-Only yields good reconstruction quality on the training frames but incorrect lighting in novel poses (baked shadows lead to unrealistic appearance).
- PBR-Only has correct relighting but lacks high-frequency fabric details (limited to rendering flat 2D textures).
- The hybrid method combines the advantages of both, outperforming each individual component on both FSIM and LPIPS.
- The fabric-specific PBR model (with sheen) significantly outperforms Lambertian and standard Disney BRDF in PSNR.
- Compared to Gaussian Garments, it saves 24.5 hours of multi-view registration time and avoids tracking-induced detail blurring.

## Highlights & Insights

- The "one-frame-only" constraint is highly practical and significantly reduces data acquisition costs.
- The high/low frequency decomposition rendering approach is elegant, fully leveraging the respective strengths of both representations.
- The fabric-specific sheen BRDF captures the key visual features of fabric appearance (stray fiber scattering).
- The reconstructed assets naturally support real-time applications (real-time PBR + real-time 3DGS rendering + real-time XPBD simulation).

## Limitations & Future Work

- It ignores the impact of deformation and novel lighting on high-frequency appearance (e.g., micro-texture changes at new wrinkles).
- The generalization capability of 3DGS in insufficiently observed regions (e.g., underarms) is limited.
- Albedo extraction relies on external models; multi-view inconsistency can cause artifacts in far-field rendering.
- It only supports thin-shell geometry assumptions, and cannot handle the interior of pockets, multi-layered clothing, or garment openings.
- Future work could restore more accurate rest shapes and material parameters via differentiable simulation to reduce the "sagging" phenomenon.

## Related Work & Insights

- The hybrid strategy of combining 3DGS with traditional PBR can be generalized to other scenarios requiring both "appearance reconstruction + physics-driven animation".
- The idea of frequency-domain decomposition can be applied to other baked lighting recovery tasks (e.g., NeRF relighting).
- The cloth BRDF model has direct value for application scenarios such as virtual try-on and fashion e-commerce.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4.5 |
| Experimental Thoroughness | 3.5 |
| Writing Quality | 4 |
| Overall Evaluation | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[CVPR 2025\] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)
- [\[CVPR 2025\] AniGS: Animatable Gaussian Avatar from a Single Image with Inconsistent Gaussian Reconstruction](anigs_animatable_gaussian_avatar_from_a_single_image_with_inconsistent_gaussian_.md)
- [\[CVPR 2026\] Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image](../../CVPR2026/3d_vision/zero-shot_reconstruction_of_animatable_3d_avatars_with_cloth_dynamics_from_a_sin.md)
- [\[CVPR 2025\] PhysAnimator: Physics-Guided Generative Cartoon Animation](physanimator_physics-guided_generative_cartoon_animation.md)

</div>

<!-- RELATED:END -->
