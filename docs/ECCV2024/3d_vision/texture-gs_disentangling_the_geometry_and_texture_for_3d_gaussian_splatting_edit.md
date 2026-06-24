---
title: >-
  [Paper Note] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This paper proposes Texture-GS, which disentangles geometry and texture for 3D Gaussian Splatting (3D-GS) for the first time. By leveraging a UV-mapping MLP and local Taylor expansion, it represents scene appearance as 2D texture maps, enabling real-time texture swapping and editing (58 FPS on an RTX 2080 Ti).
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "texture mapping"
  - "geometry-appearance disentanglement"
  - "scene editing"
  - "real-time rendering"
date: 2026-05-08
content_hash: 8e079d3116e94a4e
---

# Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing

**Conference**: ECCV 2024  
**arXiv**: [2403.10050](https://arxiv.org/abs/2403.10050)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, texture mapping, geometry-appearance disentanglement, scene editing, real-time rendering

## TL;DR

This paper proposes Texture-GS, which disentangles geometry and texture for 3D Gaussian Splatting (3D-GS) for the first time. By leveraging a UV-mapping MLP and local Taylor expansion, it represents scene appearance as 2D texture maps, enabling real-time texture swapping and editing (58 FPS on an RTX 2080 Ti).

## Background & Motivation

3D-GS stores colors in the spherical harmonics (SH) coefficients of each individual Gaussian, resulting in fully coupled geometry and appearance. Consequently, it lacks the flexibility for appearance editing (e.g., texture swapping) found in traditional mesh+texture representations. Although NeRF-based methods like NeuTex have achieved UV mapping, evaluating an MLP for every ray-Gaussian intersection is computationally expensive, hindering real-time rendering. Simply computing UV coordinates solely for Gaussian centers causes all pixels covered by the same Gaussian to map to the same UV coordinate, leading to discontinuities in the texture space.

## Method

### Overall Architecture

The method is divided into two stages. Stage 1 learns a UV-mapping MLP (constrained by cycle consistency and Chamfer distance). Stage 2 utilizes the frozen MLP to efficiently compute intersection UV coordinates via Taylor expansion and learns the 2D texture map.

### Key Designs

**UV mapping MLP learning**: First, a standard 3D-GS is trained to obtain depth maps, which are back-projected to generate a 3D point cloud of the surface. A mapping $\phi: \mathbb{R}^3 \rightarrow \mathbb{R}^2$ and its inverse mapping $\phi^{-1}$ are learned using three constraints:
- 3D Cycle Consistency: $\|x - \phi^{-1} \circ \phi(x)\|$
- 2D Cycle Consistency: $\|u - \phi \circ \phi^{-1}(u)\|$
- Chamfer Distance: The output of $\phi^{-1}$ should uniformly cover the surface point cloud

**Efficient UV mapping (Taylor expansion)**: For each Gaussian center $\mu_j$, its UV coordinates $\phi(\mu_j)$ and Jacobian matrix $J|_{\mu_j}$ are pre-computed. The UV coordinates of the intersection point are approximated using a first-order Taylor expansion: $\tilde{\phi}(I(G_j, r_p)) = \phi(\mu_j) + J|_{\mu_j}(I(G_j,r_p) - \mu_j)$. This requires only a single small matrix multiplication, ensuring real-time rendering.

**Ray-Gaussian intersection computation**: Each Gaussian is flattened along its normal vector (the eigenvector corresponding to the smallest eigenvalue) to calculate the ray-plane intersection. Opacity 0/1 regularization and normal supervision are introduced to ensure that the Gaussians are sufficiently flat.

**Color function**: The diffuse color is queried from the texture map, while the view-dependent component is represented as a per-Gaussian SH coefficient residual: $\mathcal{C}(G_j, r_p) = h(\tilde{\phi}(I(G_j,r_p)), \mathcal{T}) + c_j^{SH}$.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_{mask} + \lambda_{ssim}\mathcal{L}_{ssim} + \lambda_{01}\mathcal{L}_{01} + \lambda_n(\mathcal{L}_{norm} + \mathcal{L}_{sm}) + \lambda(\mathcal{L}_1^{noSH} + \lambda_{ssim}\mathcal{L}_{ssim}^{noSH})$$

Here, $\mathcal{L}^{noSH}$ is calculated using images rendered without SH coefficients, encouraging appearance information to be primarily stored in the texture map rather than per-Gaussian properties.

## Key Experimental Results

### Main Results on DTU Dataset

| Method | PSNR↑ | L1↓ | LPIPS↓ | FPS |
|------|-------|-----|--------|-----|
| NeuTex | 30.39 | 0.0158 | 0.1613 | 0.025 |
| Neural Gauge Fields | 29.44 | 0.0166 | 0.1506 | 0.025 |
| 3D-GS | 30.99 | 0.0121 | 0.1079 | 198 |
| **Texture-GS** | **30.03** | **0.0135** | **0.1440** | **58** |

### Ablation Study

| Method | PSNR↑ | LPIPS↓ | Description |
|------|-------|--------|------|
| Texture-GS | 30.03 | 0.1440 | Full model |
| Ours (no SH) | 27.63 | 0.1566 | Remove per-Gaussian SH |
| w/o Reg | 30.62 | 0.1374 | Remove noSH regularization |
| w/o Reg (no SH) | 25.10 | 0.1757 | Remove both |
| Pre-fetching (Center UV only) | 29.28 | 0.1557 | Naive baseline |

Effects of Gaussian Pruning:

| Gaussian Ratio | PSNR↑ | FPS |
|----------|-------|-----|
| 100% | 30.03 | 58 |
| 50% | 29.57 | 69 |
| 20% | 28.75 | 82 |
| 5% | 27.86 | 104 |

### Key Findings

- Over 2000 times faster than NeuTex (58 vs. 0.025 FPS), with training time reduced from 30 hours to 90 minutes.
- The Taylor expansion approximation is significantly faster than direct MLP evaluation, while maintaining continuous textures without artifacts.
- High-quality texture swapping results can still be synthesized even when using only 5% of the Gaussians (approx. 4.4k).
- The noSH regularization ensures that appearance information is mainly stored in the texture map, which is crucial for texture editing.

## Highlights & Insights

1. **Taylor expansion is the core innovation**: It shifts the computational complexity of the MLP from rendering time to pre-computation time, requiring only a single matrix multiplication during rendering and perfectly balancing accuracy and efficiency.
2. It bridges the gap between 3D-GS and the texture mapping concepts of traditional graphics pipelines.
3. It supports global texture swapping and local texture painting, showcasing practical use cases such as text editing.

## Limitations & Future Work

- The texture space is defined as a unit sphere, making it unsuitable for multi-object or outdoor scenes.
- Edges may appear slightly blurry during high-contrast texture swapping due to inaccurate Gaussian normals.
- The PSNR is slightly lower than that of the original 3D-GS (30.03 vs. 30.99), introducing a rendering quality loss of approximately 1dB.

## Related Work & Insights

The cycle consistency concept from NeuTex is inherited but with significantly improved efficiency. Nuvo's multi-chart design could potentially be used to extend this method to complex scenes. This work provides an important push for the 3D-GS editing ecosystem.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](../../AAAI2026/3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[CVPR 2025\] FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](../../CVPR2025/3d_vision/fruitninja_3d_object_interior_texture_generation_with_gaussian_splatting.md)
- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)
- [\[ECCV 2024\] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
