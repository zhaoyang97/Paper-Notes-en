---
title: >-
  [Paper Note] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes
description: >-
  [CVPR 2025][3D Vision][Multi-layered meshes] This paper proposes Volumetric Surfaces, a representation method that learns multi-layered translucent SDF mesh shells (k-SDF) with adaptive spacing. By rendering them via rasterization in a fixed order, it achieves real-time, high-quality view synthesis of fuzzy geometries (such as fur and hair) on low-power laptops and smartphones.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-layered meshes"
  - "volumetric surfaces"
  - "fuzzy geometry"
  - "real-time rendering"
  - "mobile rendering"
date: 2026-05-08
content_hash: 75bb989ba1077a9a
---

# Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes

**Conference**: CVPR 2025  
**arXiv**: [2409.02482](https://arxiv.org/abs/2409.02482)  
**Code**: [autonomousvision.github.io/volsurfs](https://autonomousvision.github.io/volsurfs)  
**Area**: 3D Vision / Real-time Rendering  
**Keywords**: Multi-layered meshes, volumetric surfaces, fuzzy geometry, real-time rendering, mobile rendering

## TL;DR

This paper proposes Volumetric Surfaces, a representation method that learns multi-layered translucent SDF mesh shells (k-SDF) with adaptive spacing. By rendering them via rasterization in a fixed order, it achieves real-time, high-quality view synthesis of fuzzy geometries (such as fur and hair) on low-power laptops and smartphones.

## Background & Motivation

Real-time view synthesis on mobile devices faces strict constraints in processing power, memory, and thermal dissipation. Existing methods can be categorized into two major paradigms, each with its own limitations:

1. **Surface-based methods (such as BakedSDF, MobileNeRF, BOG)**: These require only one sample point per ray and are fast, but they fail to accurately model fuzzy geometries (such as hair and fur) because all appearance information is compressed into a single surface point.
2. **Volume-based methods (such as 3DGS, SMERF)**: These represent fuzzy materials well through multi-point sampling rendering, but suffer from three performance issues—(P1) a large number (tens to hundreds) of sample points are required per ray, (P2) volume rendering requires additional data structures to skip empty space (increasing memory bandwidth), and (P3) splatting requires sorting primitives by distance (which is difficult to achieve efficiently on platforms with limited GPGPU capabilities).

Key Insight: **Textured Shells** are a classic computer graphics technique for simulating fuzzy surfaces using multiple layers of concentric, semi-transparent meshes. This paper combines this concept with differentiable rendering to learn multi-layer SDF shells with adaptive spacing, achieving: (P1) a finite and bounded number of sample points (3–9), (P2) efficient identification of sampling positions via rasterization, and (P3) fixed-order rendering without sorting.

## Method

### Overall Architecture

A two-stage training + baking pipeline: (1) Implicit Surface Stage—First, a standard NeuS model is trained to obtain the primary surface SDF, followed by initializing and training the k-SDF model (composed of $k$ adaptively spaced SDF shells + view-dependent opacity fields + color fields); (2) Baking Stage—Extracting the $k$ SDFs into lightweight meshes (using Marching Cubes + mesh simplification down to 0.02% of the original face count), generating UV atlases, training neural SH textures, and baking them into PNG images. Finally, all meshes are rasterized in a fixed order in a WebGL renderer.

### Key Designs

1. **k-SDF Representation**:
    - Function: Models $k$ surfaces as shell layers surrounding the primary SDF, ensuring that the layers do not intersect and can be traversed in a fixed order.
    - Mechanism: A primary SDF $d$ plus $k-1$ offset fields $\{o_2, ..., o_k\}$, where the SDF of each surface is $d_i = d + o_i$. Offsets are obtained by computing cumulative sums over predicted relative offsets $\hat{o}_i$ (accumulated in positive and negative directions respectively) to obtain absolute offsets, ensuring order between layers. Each surface is equipped with view-dependent opacity $\alpha(\mathbf{x}, \mathbf{v})$. The rendering formulation is a fixed-order alpha blending of $k$ surfaces: $\mathcal{R}(\mathbf{r}) = \sum_{i=1}^k \mathcal{C}_i \mathcal{A}_i w_i$
    - Design Motivation: The shell structure guarantees a fixed traversal order from outside to inside, entirely avoiding the sorting issue of 3DGS. Adaptive spacing (by learning offsets) utilizes a limited number of layers more effectively than uniform spacing.

2. **Training Strategy (Two-Stage + $\beta$ Scheduling)**:
    - Function: Ensures stable training and ultimately yields a sharp, mesh-bakeable SDF.
    - Mechanism: The first stage trains a standard NeuS model for 100k iterations, with $\beta$ exponentially scheduled from large (blurry density) to small (sharp density) to obtain a stable primary surface. The second stage uses this primary surface to initialize the k-SDF (where the remaining $k-1$ shells are distributed with uniform spacing $\Delta o = (1/\beta_2)\pi/\sqrt{3}$) and continues training for 50k iterations until the density is fully sharpened. Eikonal loss and curvature smoothing loss are applied during training to ensure SDF quality. An occupancy grid ($256^3$) is used for space skipping, and hierarchical importance sampling over multiple surfaces is employed for efficient volume rendering.
    - Design Motivation: Directly training k-SDF from scratch leads to degenerate, fully transparent solutions. Using a pretrained, opaque primary surface as an anchor for initialization prevents this issue. Initializing all shells inside the primary surface yields the best results (by increasing model capacity).

3. **Mixed-Resolution Neural Textures + Baking**:
    - Function: Converts implicit representations into lightweight explicit assets that can render in real-time on mobile devices.
    - Mechanism: Each mesh is simplified to approximately 2MB, and after generating a UV atlas, a per-surface neural SH texture is trained. The key innovation is the mixed-resolution design—the base color uses the highest resolution ($2048^2$), while high-order SH coefficients use a lower resolution ($256^2$), which significantly reduces memory (about 14MB per mesh vs. 0.5GB at full resolution). Training simulates OpenGL bilinear interpolation (predicting at texel centers and interpolating) to precisely match the training results with real-time renderers. The textures are finally baked into PNG images (using Sigmoid compression + quantization to $[0,255]$).
    - Design Motivation: Storing in full resolution is impractical, whereas spatial variations of high-order SH coefficients are typically low-frequency and can be losslessly represented at lower resolutions.

### Loss & Training

The two-stage training loss is: $\mathcal{L} = \mathcal{L}_c + \lambda_e\mathcal{L}_e + \lambda_s\mathcal{L}_s$ (with $\lambda_e=0.04$, $\lambda_s=0.65$). Here, $\mathcal{L}_c$ is the pixel-wise L1 color loss, $\mathcal{L}_e$ is the Eikonal loss (constraining the SDF gradient norm to 1), and $\mathcal{L}_s$ is the curvature smoothing loss (promoting smooth solutions for easy baking into lightweight meshes). The texture stage is trained for 15k iterations, using only the L1 color loss.

Opacity attenuation: For support surfaces, opacity is multiplied by an angular-dependent weight $\alpha_w = 2 \cdot \text{Sigmoid}(10 \cdot |\mathbf{v} \cdot \mathbf{n}|) - 1$, decaying to 0 at grazing angles to avoid hard clipping at the boundaries.

## Key Experimental Results

### Main Results (Shelly Dataset, Fuzzy Geometry)

| Method | PSNR↑ | Smartphone FPS↑ | Laptop FPS↑ | Storage (MB)↓ |
|------|-------|-------------|---------|----------|
| MobileNeRF | 29.30 | 24 | 35 | 194 |
| 3DGS-50K | 32.73 | 20 | 160 | 12 |
| 3DGS | 35.44 | 8 | 18 | 57 |
| PermutoSDF | 29.85 | - | - | - |
| **Ours (3-Mesh)** | 33.39 | **65** | 145 | 46 |
| **Ours (5-Mesh)** | 34.25 | 55 | 90 | 77 |
| **Ours (7-Mesh)** | **34.50** | 42 | 70 | 110 |
| **Ours (9-Mesh)** | 34.38 | 35 | 55 | 140 |

### Cross-Dataset Evaluation

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Custom (Furry) | 3DGS | 37.34 | 0.982 | 0.147 |
| Custom | Ours (7-Mesh) | 35.63 | 0.977 | 0.169 |
| DTU (83,105) | 3DGS | 38.06 | 0.989 | 0.086 |
| DTU | Ours (9-Mesh) | 37.17 | 0.987 | 0.083 |

### Ablation Study

| Configuration | PSNR↑ | Description |
|------|-------|------|
| 1-SDF (NeuS baseline) | ~29.85 | Single surface, fails on fuzzy geometries |
| k-SDF outer+inner initialization | Lower | Poor shell utilization |
| k-SDF all-inner initialization | Higher | Increases model capacity |
| 5-SDF vs 5-Mesh | Mesh is higher | Fixed geometry + texture optimization is more stable |
| Without opacity attenuation | Boundary hard cuts | Angle-dependent attenuation is smoother |
| Mixed-resolution vs. full $2048^2$ | Mixed is slightly better | Low-order is finer, high-order does not require high resolution |

### Key Findings

- A 7-layer mesh yields the best trade-off: quality (34.50 dB), speed (42 FPS on smartphone), and storage (110 MB) are all well-balanced.
- On smartphones, 3DGS (8 FPS) cannot achieve real-time rendering, whereas the proposed method (42 FPS) easily reaches the 30 FPS real-time standard.
- The PSNR of the proposed method is only about 1 dB lower than 3DGS (35.44 dB), but it renders more than $5\times$ faster.
- Adaptive shell spacing concentrates layers near solid structures and maintains larger gaps in fuzzy areas, automatically adapting to the scene.
- 9-layer meshes exhibit a slight performance drop because deep surface layers contribute little to pixel colors, resulting in weak gradients and slower optimization.

## Highlights & Insights

- **Precise problem definition**: It clearly identifies the three performance bottlenecks of volume rendering (sample count, space skipping, sorting) and resolves them one by one.
- **Modernization of classic computer graphics concepts**: Consolidates the concept of textured shells with differentiable rendering, learning adaptive spacing rather than uniform spacing.
- **Clever shell-constraint design in k-SDF**: Guarantees inter-layer order via cumulative sums, enabling fixed-order alpha blending without sorting.
- **Practical mixed-resolution texture strategy**: Employs band-by-band resolution for SH coefficients, yielding highly efficient storage utilization.
- **WebGL deployment**: The final assets (meshes + PNG textures) can be directly rendered in web browsers, showing excellent cross-platform compatibility.

## Limitations & Future Work

- Image quality is still approximately 1 dB lower than 3DGS, presenting a trade-off in scenarios where the highest quality is paramount.
- Scaling beyond 9 layers is hindered by vanishing gradient issues.
- The method is primarily validated on object-level scenes, and its applicability to large-scale outdoor scenes remains unexplored.
- The mesh simplification ratio is extremely aggressive (down to 0.02%), which may over-simplify certain fine-grained geometries.
- Future work could explore integration with more efficient SH representations or alternative view-dependent models.

## Related Work & Insights

- **Ours vs. 3DGS**: 3DGS achieves higher quality, but its sorting overhead prevents real-time rendering on mobile devices; the proposed sorting-free design is more than $5\times$ faster on low-end platforms.
- **Ours vs. MobileNeRF**: MobileNeRF is also targeting mobile devices but represents only a single surface, failing to handle fuzzy geometries; our multi-layer design yields a 5 dB improvement.
- **Ours vs. AdaptiveShells**: AdaptiveShells uses a single SDF + spatially-varying kernel + volume rendering; our method uses multiple SDFs + rasterized sampling, making it better suited for mobile devices.
- **Ours vs. GaussianShellMaps**: Also uses layered meshes but with fixed shell spacing and splatting, which still fails to run in real-time on low-end mobile phones.

## Rating

- Novelty: ⭐⭐⭐⭐ The k-SDF shell representation + adaptive spacing learning + mixed-resolution textures form a systematic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons against multiple baselines on several datasets, containing comprehensive 3D analysis of speed, quality, and storage.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured paper; clear problem motivation, rigorous formulation derivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Resolves the key issue of real-time rendering of fuzzy geometries on mobile devices, offering direct industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LUCAS: Layered Universal Codec Avatars](lucas_layered_universal_codec_avatars.md)
- [\[CVPR 2026\] Radiance Meshes for Volumetric Reconstruction](../../CVPR2026/3d_vision/radiance_meshes_for_volumetric_reconstruction.md)
- [\[CVPR 2025\] Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis](dynamic_neural_surfaces_for_elastic_4d_shape_representation_and_analysis.md)
- [\[CVPR 2025\] SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing](simavatar_simulation-ready_avatars_with_layered_hair_and_clothing.md)
- [\[CVPR 2025\] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos](layered_motion_fusion_lifting_motion_segmentation_to_3d_in_egocentric_videos.md)

</div>

<!-- RELATED:END -->
