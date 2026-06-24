---
title: >-
  [Paper Note] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] OceanSplat is proposed, which achieves high-fidelity underwater 3D Gaussian Splatting scene reconstruction under scattering media by incorporating trinocular view consistency constraints, synthetic epipolar depth priors, and depth-aware alpha adjustments, significantly reducing floating artifacts and outperforming existing methods.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Underwater Scene Reconstruction"
  - "Trinocular Stereo Consistency"
  - "Depth Regularization"
  - "Scattering Media"
date: 2026-05-08
content_hash: 9e4ad75b2dd2b67a
---

# OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction

**Conference**: AAAI 2026  
**arXiv**: [2601.04984](https://arxiv.org/abs/2601.04984)  
**Code**: [oceansplat.github.io](https://oceansplat.github.io)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Underwater Scene Reconstruction, Trinocular Stereo Consistency, Depth Regularization, Scattering Media

## TL;DR

OceanSplat is proposed, which achieves high-fidelity underwater 3D Gaussian Splatting scene reconstruction under scattering media by incorporating trinocular view consistency constraints, synthetic epipolar depth priors, and depth-aware alpha adjustments, significantly reducing floating artifacts and outperforming existing methods.

## Background & Motivation

Underwater scene reconstruction is essential for marine robotics tasks such as seafloor mapping, ecological monitoring, and underwater infrastructure inspection. However, the optical properties of underwater environments (wavelength-dependent attenuation, scattering, and low illumination) severely degrade visual cues, posing significant challenges to vision-based scene reconstruction.

**Limitations of Prior Work**:

**NeRF Methods** (e.g., SeaThru-NeRF): Embed underwater physical models into volume rendering; however, implicit representations hinder accurate geometric understanding, and rendering speed is slow.

**3DGS Methods** (e.g., SeaSplat, WaterSplatting): While rendering is fast, the **medium intensity is often absorbed into the 3D Gaussians**, leading to a large number of floating artifacts. The entanglement of 3D Gaussians and the scattering medium degrades reconstruction quality.

**Core Problem**: In scattering media, view-dependent sampling of alpha-blending causes multi-view inconsistency. Consequently, 3D Gaussians tend to falsely represent the water body itself rather than the scene objects, producing floating artifacts.

**Mechanism**:
- Drawing on the principle that multi-baseline stereo vision outperforms single-baseline, binocular consistency is extended to **trinocular consistency** (horizontal and vertical virtual viewpoints) to provide orthogonal constraints.
- **Self-supervised depth priors** are generated using triangulation between virtual viewpoints.
- **Depth-aware alpha adjustments** are employed to suppress 3D Gaussians in medium regions during the early stages of training.

## Method

### Overall Architecture

OceanSplat is based on the 3DGS framework, initializing 3D Gaussians using SfM and modeling underwater medium properties (attenuation, backscattering, medium color) via an MLP. Four key modules are introduced during training: trinocular view consistency, synthetic epipolar depth prior, depth residual loss, and depth-aware alpha adjustment.

The underwater image formation model decomposes the observed image into attenuated object color and backscattering:
$$C = C^{obj} \cdot e^{-\sigma^{attn} \cdot z} + C^{\infty} \cdot (1 - e^{-\sigma^{bs} \cdot z})$$

Object and medium rendering are accumulated separately via alpha-blending, which supports object-medium decoupling.

### Key Designs

1. **Trinocular View Consistency**

   **Mechanism**: Two virtual viewpoints, horizontal $P_h$ and vertical $P_v$, are generated from the original camera pose $P_c$. Consistency among the three views is enforced to regularize the spatial positions of the 3D Gaussians.

   The virtual viewpoints are constructed via translation:
    $P_h = \begin{bmatrix} \mathbb{I} & \mathbf{t}_h \\ \mathbf{0}^\top & 1 \end{bmatrix} P_c, \quad P_v = \begin{bmatrix} \mathbb{I} & \mathbf{t}_v \\ \mathbf{0}^\top & 1 \end{bmatrix} P_c$
   where $\mathbf{t}_h = (b_h, 0, 0)^\top$ and $\mathbf{t}_v = (0, b_v, 0)^\top$.

   After rendering images from virtual viewpoints, disparities are calculated using the depth map to perform backward warping, aligning the virtual viewpoint images with the center view:
    $d_h(x,y) = \frac{f_h \cdot b_h}{D_c(x,y)}, \quad d_v(x,y) = \frac{f_v \cdot b_v}{D_c(x,y)}$

   The consistency loss consists of three parts:
    - **Object Stereo Consistency**: $L_{obj\text{-}stereo}$, the R-L1 loss between the warped object image and the center view object image.
    - **Full Stereo Consistency**: $L_{full\text{-}stereo}$, the R-L1 loss between the synthesized full image and the ground truth (GT).
    - **Disparity Smoothness**: $L_{smooth}$, the edge-aware disparity regularization.

   **Design Motivation**: A single-baseline stereo workflow only provides constraints in one direction. Horizontal and vertical orthogonal baselines provide stronger spatial constraints, which better resolve geometric ambiguity in scattering media. $b_v$ is sampled from [-0.4, 0.4], and $b_h = 1.5 b_v$, using unequal baselines to increase constraint diversity.

2. **Synthetic Epipolar Depth Prior**

   **Mechanism**: A self-supervised depth prior $D_{epi}$ is derived using triangulation between virtual viewpoints, eliminating the need for external depth supervision.

   Specific steps:
    - Select 3D Gaussians with opacity $> \tau_\alpha$ within the intersection of the trinocular viewing frustums.
    - Project the selected Gaussians onto the image planes of $P_h$ and $P_v$.
    - Establish a linear system $\mathbf{A}_i\tilde{\mathbf{X}}_i = \mathbf{0}$ through epipolar geometry.
    - Solve the triangulated points via least squares, and transform them to the central camera coordinate system, taking the z-component as the depth prior.

   Apply the edge-aware Log-L1 loss:
    $L_{epi} = \frac{1}{HW}\sum_{x,y}\sum_{k}\log(1 + |D_c' - D_{epi}|) \cdot e^{-|\nabla_k I_c|}$

   **Design Motivation**: Geometric cues are limited in underwater scenes, and external depth models can be inaccurate. Utilizing the geometric relationships of internal virtual viewpoints provides self-consistent depth constraints, avoiding external dependencies.

3. **Depth Residual Loss**

   Constrains the z-component of each 3D Gaussian to be consistent with the alpha-blended rendered depth:
    $L_{res} = \frac{1}{N'}\sum_{i=1}^{N'}|D_c(\mathbf{x}_i) - z_i|$

   This prevents 3D Gaussians from dispersing excessively along the ray, thereby reducing floating artifacts.

4. **Depth-aware Alpha Adjustment**

   In the early stages of training ($t < t_\alpha$), an MLP is used to adjust the opacity of each 3D Gaussian based on its depth and viewing direction:
    $\alpha_i' = (1-w)\alpha_i + w \cdot \phi_\alpha(\alpha_i, z_i, \vec{\mathbf{v}}_i)$

   After the transition step $t_\alpha$, the weight $w$ decays to 0, eliminating any inference overhead.

   **Design Motivation**: In scattering media, mispositioned 3D Gaussians erroneously gain contributions from the medium color. By suppressing the opacity of these Gaussians early in training, they are encouraged to be pruned, thereby preventing medium-induced artifacts at the source.

### Loss & Training

$$L_{total} = L_{photo} + \lambda_{tri} L_{tri} + \lambda_{epi} L_{epi} + \lambda_{res} L_{res}$$

- $L_{photo}$: Weighted R-L1 + R-SSIM ($\lambda_s = 0.2$)
- $\lambda_{tri} = 0.1$, $\lambda_{res} = 0.01$
- $\lambda_{epi}$ is annealed from 0.4 to 0.2
- Training steps: SeaThru-NeRF data uses 7K/3K steps (densification/fine-tuning); In-the-Wild data uses 10K/5K steps
- Progressive resolution training: 1/4 -> 1/2 -> full resolution

## Key Experimental Results

### Main Results

**Real-world Underwater Scenes (SeaThru-NeRF + In-the-Wild)**:

| Dataset | Metric | OceanSplat | WaterSplatting | SeaSplat | Gain |
|--------|------|------------|----------------|----------|------|
| Curaçao | PSNR | **34.56** | 32.32 | 29.77 | +2.24 |
| Panama | PSNR | **32.74** | 31.71 | 28.65 | +1.03 |
| J.G-Redsea | PSNR | **25.35** | 24.77 | 23.07 | +0.58 |
| IUI3-Redsea | PSNR | **30.17** | 29.84 | 27.23 | +0.33 |
| Coral | PSNR | **29.15** | 28.19 | 28.41 | +0.96 |
| Composite | PSNR | 26.39 | 25.47 | 26.22 | +0.92 |

The average PSNR outperforms WaterSplatting by 1.05 dB and exceeds SeaThru-NeRF-NS by 2.88 dB.

**Simulated Scattering Scenes (Underwater + Fog)**:

| Scene | Metric | OceanSplat | WaterSplatting | SeaSplat |
|------|------|------------|----------------|----------|
| Underwater-NVS | PSNR | **28.80** | 28.12 | 15.62 |
| Fog-NVS | PSNR | **29.12** | 28.45 | 27.52 |
| Underwater-Restoration | SSIM | **0.768** | 0.748 | 0.719 |
| Fog-Restoration | SSIM | **0.791** | 0.770 | 0.744 |

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS | Description |
|------|------|------|-------|------|
| Full Model | **34.56** | **0.961** | **0.113** | Full model |
| w/o $L_{res}$ | 34.30 | 0.960 | 0.115 | Depth residual loss is effective |
| w/o $L_{epi}$ | 33.82 | 0.959 | 0.120 | Epipolar depth prior provides a significant contribution |
| w/o $L_{tri}$ | 33.20 | 0.957 | 0.115 | Trinocular consistency contributes the most (-1.36 dB) |
| w/o $\alpha^d$ | 33.90 | 0.960 | 0.116 | Depth-aware alpha adjustment is effective |

Efficiency comparison: Training takes 19 minutes (vs 18h 25m for SeaThru-NeRF), rendering runs at 85.67 FPS, and VRAM usage is 7.6 GB.

### Key Findings

- Trinocular consistency is the most critical component (removing it decreases the PSNR by 1.36 dB).
- The contribution of the epipolar depth prior ranks second (-0.74 dB).
- Depth-aware alpha adjustment is highly effective in suppressing medium artifacts.
- All referenced components are designed in a self-supervised manner, requiring no external depth GT or annotations.

## Highlights & Insights

1. **Clear geometric motivation for the trinocular extension**: Compared to binocular methods which only constrain horizontally, adding a vertical virtual viewpoint introduces constraints in orthogonal directions, supported by a solid foundation in stereo geometry.
2. **Fully self-supervised depth regularization**: The synthetic epipolar depth prior is derived from the model's own virtual viewpoint triangulation, without relying on any external depth models, achieving a "self-consistent" geometric constraint.
3. **Object-medium decoupling**: Effective geometric constraints promote the separation of 3D Gaussians from the scattering medium. This both improves reconstruction quality and enables scene restoration (e.g., water/fog removal).
4. **"Preventative" strategy via early alpha adjustments**: Rather than correcting artifacts after they appear, the method proactively suppresses problematic 3D Gaussians during the early stages of training.

## Limitations & Future Work

- Each iteration requires additional rasterization (rendering for virtual viewpoints) and least-squares solving, resulting in a slightly longer training time compared to WaterSplatting (19 min vs 10 min).
- The virtual viewpoint baseline lengths $b_h, $b_v$ are empirical values and may be sensitive to different scene scales.
- The approach is currently validated only on static underwater scenes; dynamic elements (e.g., water currents, bubbles) are not addressed.
- The scattering model remains simplified and does not fully capture complex wavelength-dependent scattering effects.

## Related Work & Insights

- **WaterSplatting (2024)**: A hybrid approach using an implicit medium + explicit objects; it serves as the main competitor and baseline in this work.
- **SeaSplat (2024)**: Integrates underwater physics into 3D Gaussians, but suffers from insufficient geometric constraints.
- **StereoGS (2024, Han et al.)**: Uses binocular stereo consistency to regularize 3DGS; the proposed work extends this concept to a trinocular setup.
- **Insight**: The concept of constructing constraints via virtual viewpoints can be generalized to the 3D reconstruction of other degraded scenes (e.g., fog, smoke, dust).

## Rating

- Novelty: ⭐⭐⭐⭐ (Well-motivated trinocular extension, ingenious design of self-supervised depth priors)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Real + simulation, NVS + restoration, complete ablation study, comprehensive efficiency comparison)
- Writing Quality: ⭐⭐⭐⭐⭐ (Complete mathematical derivations, clear illustrations, sufficient physical motivation explanation)
- Value: ⭐⭐⭐⭐ (A significant advancement in underwater scene reconstruction, offering practical self-supervised design)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Uncertainty-Aware 3D Reconstruction for Dynamic Underwater Scenes](../../ICLR2026/3d_vision/uncertainty-aware_3d_reconstruction_for_dynamic_underwater_scenes.md)
- [\[ICLR 2026\] Signal Structure-Aware Gaussian Splatting for Large-Scale Scene Reconstruction](../../ICLR2026/3d_vision/signal_structure-aware_gaussian_splatting_for_large-scale_scene_reconstruction.md)
- [\[CVPR 2026\] Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting](../../CVPR2026/3d_vision/intrinsic_geometry-appearance_consistency_optimization_for_sparse-view_gaussian_.md)
- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[AAAI 2026\] Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction](splat-sap_feed-forward_gaussian_splatting_for_human-centered_scene_with_scale-aw.md)

</div>

<!-- RELATED:END -->
