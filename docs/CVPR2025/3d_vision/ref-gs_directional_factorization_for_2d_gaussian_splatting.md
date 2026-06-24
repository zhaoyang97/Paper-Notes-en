---
title: >-
  [Paper Note] Ref-GS: Directional Factorization for 2D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][2D Gaussian Splatting] This paper proposes Ref-GS, which introduces deferred rendering and directional factorization into 2D Gaussian Splatting (2DGS). It models far-field illumination and surface roughness variations using a Sph-Mip spherical feature grid, and then achieves spatially varying view-dependent effects via compact tensor decomposition. This approach achieves state-of-the-art (SOTA) performance in reflective scene rendering and geometry reco…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "2D Gaussian Splatting"
  - "Reflection Modeling"
  - "Deferred Rendering"
  - "Directional Factorization"
  - "Normal Recovery"
date: 2026-05-08
content_hash: dd799d510c9e09ad
---

# Ref-GS: Directional Factorization for 2D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2412.00905](https://arxiv.org/abs/2412.00905)  
**Code**: [Project Page](https://ref-gs.github.io/)  
**Area**: 3D Vision  
**Keywords**: 2D Gaussian Splatting, Reflection Modeling, Deferred Rendering, Directional Factorization, Normal Recovery

## TL;DR

This paper proposes Ref-GS, which introduces deferred rendering and directional factorization into 2D Gaussian Splatting (2DGS). It models far-field illumination and surface roughness variations using a Sph-Mip spherical feature grid, and then achieves spatially varying view-dependent effects via compact tensor decomposition. This approach achieves state-of-the-art (SOTA) performance in reflective scene rendering and geometry recovery while maintaining real-time rendering at over 45 FPS.

## Background & Motivation

- **Importance of View-Dependent Effects**: Reflection and refraction are crucial elements for photorealistic rendering. However, NeRF and 3DGS model view-dependent color using Spherical Harmonics (SH), which essentially assumes that each point emits radiation independently, failing to correctly handle light bounces.
- **Ambiguity in Direction Query for 3DGS**: Ref-NeRF replaces the view direction with the reflection direction for color queries. However, a direct application in GS leads to ambiguity between SH coefficients and primitive orientations—transforming the reflection direction can be compensated for by updates in the SH coefficients.
- **Redundancy in Forward Rendering**: Forward rendering in 3DGS computes lighting for each primitive independently before projecting them to the screen, which results in significant redundant shading computations in scenes with high depth complexity.
- **Insufficient Modeling of Near-Field Illumination**: Existing methods either assume all light sources are at infinity (far-field) using a global environment map, or only model direct illumination, rendering them unable to handle near-field cases where light sources or objects are close to the target surface.
- **Goal**: To introduce high-fidelity rendering of view-dependent effects (reflection, refraction, specular highlights) while preserving the accurate geometric reconstruction capability of 2DGS.

## Method

### Overall Architecture

Ref-GS adopts a deferred rendering architecture, which is split into a geometry pass and a lighting pass. Geometry pass: 2DGS primitives blend attributes (appearance features $\mathbf{K}$, roughness $\mathbf{M}$, normals $\mathbf{N}$) into a G-buffer via alpha blending. Lighting pass: The reflection direction $\omega_r$ is computed based on the G-buffer and encoded via Sph-Mip to obtain directional features $\mathbf{s}$. Rendering pass: The spatial and directional features are connected through the outer product of tensor decomposition $\mathbf{s} \circ \mathbf{k}$, which is then decoded by an MLP to yield the final color $\mathbf{I} = \mathbf{I}_d + f_\Theta(\mathbf{S}, \mathbf{K} \otimes \mathbf{S})$.

### Key Designs

**Design 1: Deferred Gaussian Shading**
- **Function**: Eliminate the ambiguity of primitive-level direction querying in forward rendering.
- **Mechanism**: Instead of computing view-dependent colors on each primitive independently, primitive attributes (diffuse color $\mathbf{c}_d$, features $\mathbf{f}$, roughness $\rho$) are first blended into a G-buffer via standard alpha blending. Lighting computations are then performed after obtaining the expected attributes for each pixel. Color is decomposed into a diffuse component $\mathbf{I}_d$ and a specular component computed by a shader $f_\Theta$.
- **Design Motivation**: In forward rendering, each primitive queries the reflection direction independently, causing ambiguity between SH coefficients and orientation. Deferred rendering queries on the blended surface, thereby eliminating this ambiguity (as shown in Fig. 3(c)).

**Design 2: Sph-Mip Spherical Multi-scale Feature Grid**
- **Function**: Model far-field high-frequency illumination and perceive surface roughness.
- **Mechanism**: Feature points are distributed on a sphere and unfolded into a 2D feature grid using a latitude-longitude layout. The reflection direction $\omega_r$ is converted into spherical coordinates $(\theta, \phi)$, which, together with the roughness $\rho$, undergo trilinear interpolation on a 3D grid $(\theta, \phi, \rho)$ to obtain directional features $\mathbf{s} = \text{Sph-Mip}(\omega_r, \rho, \mathcal{M})$. Within the multi-scale mipmap structure, the base level $\mathcal{M}^{L_0}$ has the highest resolution, and subsequent levels halve their resolution step-by-step. Higher roughness maps to lower-level (coarser) features.
- **Design Motivation**: SH cannot represent high-frequency environmental illumination. The mipmap structure naturally maps to the physical meaning of roughness—smooth surfaces produce sharp reflections (high resolution), while rough surfaces produce blurry reflections (low resolution).

**Design 3: Spatial-Directional Factorization via Tensor Decomposition**
- **Function**: Efficiently represent spatially varying view-dependent effects.
- **Mechanism**: Spatial features $\mathbf{k} \in \mathbb{R}^D$ and directional features $\mathbf{s} \in \mathbb{R}^C$ generate a $D \times C$ matrix through a vector outer product, which is flattened and fed into a lightweight MLP to decode the final color. This utilizes a low-rank tensor decomposition inspired by TensoRF: $\mathbf{I} = \mathbf{I}_d + f_\Theta(\mathbf{S}, \mathbf{K} \otimes \mathbf{S})$.
- **Design Motivation**: The outer product factorization decouples geometry and lighting into independent vectors, avoiding the storage of high-dimensional features on each primitive (which reduces volume rendering overhead). Meanwhile, it preserves the interaction between spatially varying material properties and directionally varying lighting.

### Loss & Training

The total loss is formulated as $\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_n \mathcal{L}_{\text{normal}} + \lambda_d \mathcal{L}_{\text{depth}}$, which comprises L1 + D-SSIM color reconstruction loss, normal consistency regularization, and depth regularization.

## Key Experimental Results

### Main Results: Shiny Blender Dataset PSNR↑

| Method | Car | Ball | Helmet | Toaster | Avg. |
|------|-----|------|--------|---------|------|
| Ref-NeRF | 30.41 | 29.14 | 29.92 | 25.29 | 32.32 |
| 3DGS | 27.24 | 27.69 | 28.32 | 20.99 | 30.37 |
| GaussianShader | 27.51 | 29.02 | 28.73 | 22.86 | 30.42 |
| 3DGS-DR | 30.43 | 33.44 | 31.49 | 26.69 | 33.94 |
| **Ref-GS** | **30.94** | **36.10** | **33.40** | **27.28** | **34.80** |

### Shiny Real Dataset

| Method | Garden | Sedan | Toycar | Avg. |
|------|--------|-------|--------|------|
| Ref-NeRF | 22.01 | 25.21 | 23.65 | 23.62 |
| 3DGS | 21.75 | 26.03 | 23.78 | 23.85 |
| 3DGS-DR | 21.52 | 26.32 | 23.57 | 23.80 |
| **Ref-GS** | **22.48** | **26.63** | **24.20** | **24.44** |

### Key Findings

1. Ref-GS achieves an average PSNR of 34.80 on Shiny Blender, outperforming 3DGS-DR (33.94) and surpassing implicit methods such as ENVIDR (32.88) and Ref-NeRF (32.32).
2. It delivers rendering speeds of > 45 FPS at $800 \times 800$ resolution, maintaining real-time performance.
3. Deferred rendering effectively eliminates the direction query ambiguity—successfully reconstructing the geometry of the reflective tabletop in the Garden scene, where existing methods failed.
4. The quality of normal recovery is significantly superior to existing GS methods, particularly showing outstanding performance on specular objects (Toaster, Bell).

## Highlights & Insights

- **Adopting the deferred rendering approach from computer graphics** solves the direction ambiguity issue inherent to GS in a simple yet effective manner.
- **Sph-Mip grid** elegantly models roughness as mipmap level selection, providing clear physical intuition.
- **The outer-product factorization** simultaneously reduces the feature dimension per primitive and the volume rendering overhead, yielding a double-win design.
- It achieves an excellent balance between pursuing SOTA rendering quality and accurate geometric recovery simultaneously.

## Limitations & Future Work

- Sph-Mip primarily models far-field illumination, still showing limitations for complex near-field lighting (e.g., inter-reflections between objects or indirect reflections caused by self-occlusion).
- Deferred rendering assumes a single surface per pixel, offering limited capability when handling translucent or multi-layer refractive materials.
- It requires known camera parameters and SfM initialization, making it unable to handle scenes with unposed inputs.
- The spherical Mip-grid is a single, globally shared environment map, which cannot learn independent local environmental illumination for different regions of a scene.
- The generalization capability in large-scale outdoor scenes remains to be further validated.

## Related Work & Insights

- **vs GaussianShader**: GaussianShader models view-dependent effects separately but still operates at the primitive level, suffering from direction ambiguity. Ref-GS fundamentally resolves this issue by deferring computation to the pixel level.
- **vs 3DGS-DR**: 3DGS-DR also introduces deferred rendering for reflection modeling. However, Ref-GS incorporates Sph-Mip encoding and tensor decomposition, outperforming it in high-frequency reflection modeling quality and near-field lighting handling.
- **vs Ref-NeRF**: Ref-NeRF uses integrated directional encoding, which achieves good results in continuous NeRF representations but suffers from slow training and rendering. Ref-GS transfers similar concepts to GS and resolves the ambiguity issue of discrete representations via deferred shading, while maintaining real-time rendering at >45 FPS.
- The deferred rendering + G-buffer design concept of Ref-GS can be extended to other GS applications requiring material decomposition (e.g., scene editing, relighting).

## Rating

⭐⭐⭐⭐ — Elegantly introduces deferred rendering from computer graphics into 2DGS, achieving high-quality reflection rendering and accurate geometric recovery simultaneously. Both the Sph-Mip and tensor decomposition designs provide independent and valuable contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [\[NeurIPS 2025\] Anti-Aliased 2D Gaussian Splatting](../../NeurIPS2025/3d_vision/anti-aliased_2d_gaussian_splatting.md)
- [\[CVPR 2025\] HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)

</div>

<!-- RELATED:END -->
