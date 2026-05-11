---
title: >-
  [Paper Note] PhyGaP: Physically-Grounded Gaussians with Polarization Cues
description: >-
  [CVPR2026][3D Vision][3D Gaussian Splatting] This paper proposes PhyGaP, which integrates polarization cues into 2DGS optimization via a polarization deferred rendering pipeline (PolarDR)…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "polarization imaging"
  - "inverse rendering"
  - "relighting"
  - "reflection decomposition"
  - "pBRDF"
  - "environment lighting"
date: 2026-05-08
content_hash: 7f03e81784b7951d
---

# PhyGaP: Physically-Grounded Gaussians with Polarization Cues

**Conference**: CVPR2026
**arXiv**: [2603.14001](https://arxiv.org/abs/2603.14001)
**Code**: Coming soon
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, polarization imaging, inverse rendering, relighting, reflection decomposition, pBRDF, environment lighting

## TL;DR

This paper proposes PhyGaP, which integrates polarization cues into 2DGS optimization via a polarization deferred rendering pipeline (PolarDR), and introduces a self-occlusion-aware GridMap environment representation, enabling accurate reflection decomposition and realistic relighting of glossy objects.

## Background & Motivation

1. **Difficulty in reconstructing reflective objects**: 3DGS and its variants lack explicit geometric representations, and the splatting pipeline cannot simulate secondary light transport, limiting their ability to model glossy surfaces.
2. **Insufficient information in RGB images**: Existing DR methods rely on accurate estimation of normals, reflectance, and roughness, yet ordinary RGB images do not encode these physical properties, leading to failure in decomposing albedo from specular reflection.
3. **Poor relighting quality**: Due to inaccurate reflection decomposition, existing methods frequently exhibit color shifts, unrealistic shadows, or surface discontinuities under novel illumination conditions.
4. **Rich physical information encoded in polarization**: Specular reflection produces strong linear polarization, while diffuse reflection produces weak polarization with a 90° shift in polarization angle, making polarization cues naturally suited for guiding reflectance property learning.
5. **Self-occlusion in non-convex objects**: Environment cubemaps assume light sources at infinity and cannot handle self-occlusion or indirect illumination of non-convex objects, causing relighting artifacts.
6. **Existing polarization methods do not support relighting**: PANDORA encodes environment maps implicitly, and PolGS does not decompose albedo; neither supports illumination replacement.

## Method

### Overall Architecture

PhyGaP builds upon the 2DGS + Ref-Gaussian framework. Each Gaussian primitive maintains learnable attributes: albedo $\boldsymbol{\lambda}$, index of refraction (IoR) $\eta$, surface normal $\mathbf{n}$, roughness $r$, and a learnable environment cubemap mipmap $E$. These attributes are splatted into 2D material maps via α-blending and then fed into PolarDR to compute per-pixel Stokes vectors, which are supervised against ground-truth polarization observations.

### PolarDR: Polarization Deferred Rendering

- The polarization state of light is represented by the Stokes vector $\mathbf{s}=[s_0, s_1, s_2, s_3]^\top$; light–surface interactions are modeled using Mueller matrices.
- **Specular polarization**: The degree of polarization $\beta_s$ is computed from Fresnel coefficients $R^\perp, R^\parallel$ and combined with specular radiance $L_s$ to obtain the specular Stokes component.
- **Diffuse polarization**: The degree of polarization $\beta_d$ is computed from transmittance Fresnel coefficients $T^\perp, T^\parallel$ and combined with diffuse radiance $L_d$ to obtain the diffuse Stokes component.
- The sum of both components constitutes the rendered Stokes vector, which is directly compared with ground-truth polarization maps to explicitly constrain specular–diffuse decomposition.
- **Spherical harmonics are not used** to represent color, as albedo should be view-independent.

### GridMap: Self-Occlusion-Aware Environment Map

- The bounding box of the object is divided into a 3×3 grid on each face, with anchor cameras placed at grid nodes (excluding the bottom face), yielding $N=52$ anchor cameras in total.
- A single-bounce ray tracing step is performed at each anchor camera to construct a local cubemap $\tilde{E}_i$ that blends the object's own color with the global environment.
- During rendering, the Stokes contributions from all local cubemaps are fused via distance-weighted blending:

$$\tilde{S}_d = \frac{\sum_{i=1}^{N} \|\mathbf{p}-\mathbf{c}_i\|_2 \cdot \tilde{S}_d^{(i)}}{\sum_{i=1}^{N} \|\mathbf{p}-\mathbf{c}_i\|_2}$$

- Local cubemaps require no gradient computation and need only low-frequency updates, making their overhead far lower than multi-bounce ray tracing.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\mathrm{rgb}} + \lambda_1 \mathcal{L}_{\mathrm{pol}} + \lambda_2 \mathcal{L}_{\mathrm{mask}} + \lambda_3 \mathcal{L}_{\mathrm{depth}} + \lambda_4 \mathcal{L}_{\mathrm{smooth}}$$

| Loss Term | Function |
|---|---|
| $\mathcal{L}_{\mathrm{rgb}}$ | 0.8 L1 + 0.2 DSSIM for RGB reconstruction |
| $\mathcal{L}_{\mathrm{pol}}$ | L1 loss on $s_1, s_2$ for polarization reconstruction |
| $\mathcal{L}_{\mathrm{mask}}$ | Segmentation mask supervision to eliminate floater Gaussians |
| $\mathcal{L}_{\mathrm{depth}}$ | Depth–normal consistency to constrain 2DGS surface alignment |
| $\mathcal{L}_{\mathrm{smooth}}$ | Edge-aware normal smoothing to regularize normal variation |

## Key Experimental Results

### Novel View Synthesis and Normal Reconstruction

Evaluation on 9 scenes from the PANDORA/RMVP/SMVP/Mitsuba3 datasets. PhyGaP achieves an average improvement of approximately **2 dB PSNR** over RGB-only methods and reduces normal cosine distance by **45.7%**.

| Method | owl PSNR↑ | frog PSNR↑ | dog PSNR↑ | teapot PSNR↑ | frog CD↓ | dog CD↓ | teapot CD↓ |
|---|---|---|---|---|---|---|---|
| Ref-Gaussian | 22.39 | 34.13 | 37.94 | 29.67 | 0.1122 | 0.0207 | 0.0093 |
| 3DGS-DR | 24.20 | 34.68 | 39.59 | 29.07 | 0.0484 | 0.0462 | 0.0325 |
| PolGS | 24.99 | 28.25 | 28.15 | - | 0.0343 | 0.0297 | - |
| **PhyGaP** | **28.14** | **32.92** | **37.82** | **29.69** | **0.0482** | **0.0261** | **0.0079** |

### Relighting Evaluation

| Method | Env. Map PSNR (teapot)↑ | Env. Map PSNR (matpre.)↑ | Relighting PSNR↑ | Relighting SSIM↑ | Relighting LPIPS↓ |
|---|---|---|---|---|---|
| GIR | 10.30 | 10.73 | 18.02 | 0.960 | 0.0327 |
| **PhyGaP** | **11.50** | **17.46** | **19.18** | **0.973** | **0.0255** |

### Ablation Study

| Configuration | Relighting PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| w/o PolarDR & w/o GridMap | 15.56 | 0.955 | 0.0369 |
| PolarDR only (w/o GridMap) | 17.81 | 0.967 | 0.0321 |
| **Full PhyGaP** | **19.18** | **0.973** | **0.0255** |

- PolarDR effectively prevents specular reflection from contaminating albedo estimation, improving environment map quality.
- GridMap resolves self-occlusion shadows on non-convex geometry, recovering consistent surface color.

## Highlights & Insights

- **First polarization GS method with relighting capability**: While PANDORA, PolGS, and other polarization-based methods do not support relighting, PhyGaP achieves explicit reflection decomposition and illumination replacement.
- **Physics-driven polarization rendering**: PolarDR embeds the pBRDF model into GS deferred rendering and directly supervises with polarization Stokes vectors, eliminating the albedo–illumination ambiguity.
- **Practical and efficient GridMap**: The combination of 52 anchor cameras and distance-weighted fusion addresses indirect illumination without scene-specific parameters, with manageable overhead and straightforward GPU parallelization.
- **Compatible with partial polarization inputs**: Data acquisition requires only two standard RGB cameras equipped with linear polarizers, without reliance on dedicated polarization cameras.

## Limitations & Future Work

- **Insufficient modeling of metallic surfaces**: The pBRDF of metals involves complex-valued indices of refraction and phase terms that the current model may not accurately capture.
- **GridMap limitations for extreme geometries**: Highly irregular objects or strongly specular scenes with multiple inter-reflections remain challenging.
- **Infinite light source assumption in environment maps**: Finite-distance light sources in real scenes introduce reconstruction bias.
- **Multi-bounce light transport not modeled**: GridMap performs only single-bounce ray tracing; complex inter-reflection scenarios leave room for improvement.

## Related Work & Insights

| Method | Representation | Polarization | Reflection Decomp. | Relighting | Indirect Illumination |
|---|---|---|---|---|---|
| Ref-Gaussian | 2DGS+DR | ✗ | Partial | ✗ | Learned SH |
| 3DGS-DR | 3DGS+DR | ✗ | Partial | ✗ | - |
| PANDORA | NeRF | ✓ | ✓ | ✗ | Implicit |
| PolGS | 3DGS | ✓ | Partial (no albedo) | ✗ | - |
| GIR | 3DGS+DR | ✗ | ✓ | ✓ | - |
| **PhyGaP** | **2DGS+PolarDR** | **✓** | **✓ (complete)** | **✓** | **GridMap** |

## Rating

- Novelty: ⭐⭐⭐⭐ — Integrating polarization pBRDF into GS deferred rendering alongside the GridMap design for indirect illumination represents a novel technical combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 9 scenes, both synthetic and real data, and evaluates NVS, normals, decomposition, and relighting with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with complete formula derivations and rich figures and tables.
- Value: ⭐⭐⭐⭐ — First to achieve relighting capability in polarization-based GS, with practical application prospects in VR/AR and interactive design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](../../ICCV2025/3d_vision/geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)
- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)
- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)
- [\[CVPR 2026\] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI](wanderland_geometrically_grounded_simulation_for_open-world_embodied_ai.md)

</div>

<!-- RELATED:END -->
