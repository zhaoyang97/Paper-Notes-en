---
title: >-
  [Paper Note] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis
description: >-
  [ICCV 2025][3D Vision][Gaussian Splatting] This paper proposes BBSplat, which replaces the Gaussian opacity in 2D Gaussian Splatting with learnable RGB texture maps and alpha maps, enabling each planar primitive to possess arbitrary shape and per-pixel color control. With fewer primitives, BBSplat closes the rendering quality gap between 2DGS and 3DGS while preserving accurate mesh extraction capability and achieving up to ×17 storage compression.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Gaussian Splatting"
  - "textured primitives"
  - "novel view synthesis"
  - "model compression"
  - "mesh extraction"
date: 2026-05-08
content_hash: f3b377c642d31f52
---

# BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis

**Conference**: ICCV 2025
**arXiv**: [2411.08508](https://arxiv.org/abs/2411.08508)  
**Code**: [GitHub](https://github.com/david-svitov/BBSplat)  
**Area**: 3D Vision / Novel View Synthesis / 3D Scene Representation
**Keywords**: Gaussian Splatting, textured primitives, novel view synthesis, model compression, mesh extraction

## TL;DR

This paper proposes BBSplat, which replaces the Gaussian opacity in 2D Gaussian Splatting with learnable RGB texture maps and alpha maps, enabling each planar primitive to possess arbitrary shape and per-pixel color control. With fewer primitives, BBSplat closes the rendering quality gap between 2DGS and 3DGS while preserving accurate mesh extraction capability and achieving up to ×17 storage compression.

## Background & Motivation

- **3DGS** employs 3D ellipsoids with Gaussian opacity, yielding high rendering quality but precluding accurate mesh extraction.
- **2DGS** uses 2D flat ellipses aligned with object surfaces for more accurate mesh extraction, but its rendering quality lags behind 3DGS.
- Conventional Gaussian primitives (both 2D and 3D) assign a single global color (determined by spherical harmonics) and Gaussian-distributed opacity to each primitive, requiring a large number of primitives to represent high-frequency details.
- **Billboard Cloud** (a classical CG technique, Décoret et al. 2003): simplifies 3D models using sets of textured planes, widely used in real-time forest rendering and similar applications.

**Core Insight**: Equipping 2D planar primitives with learnable textures (color + opacity) enables each primitive to represent complex shapes and color variations, substantially reducing the required number of primitives while retaining the mesh extraction advantages of 2DGS.

## Core Problem

How to design a 3D scene representation that simultaneously achieves the following three objectives:
1. Match or surpass 3DGS rendering quality (closing the quality gap of 2DGS).
2. Preserve the accurate mesh extraction capability of 2DGS.
3. Significantly reduce storage footprint.

## Method

### Overall Architecture

COLMAP point cloud and camera poses → initialize billboard primitives → end-to-end optimization via photometric loss + texture regularization → MCMC adaptive density control → compressed texture output.

Each billboard is parameterized as $\{\mu_i, s_i, r_i, \text{SH}_i, T_i^{\text{RGB}}, T_i^{\alpha}\}$, where the first four terms are inherited from 2DGS (position, 2D scale, rotation quaternion, spherical harmonic coefficients), and the latter two are newly introduced RGB texture maps and alpha maps (default size $16 \times 16$).

### Key Designs

1. **Textured Planar Primitives**:

    - The **alpha map $T_i^{\alpha}$** replaces the global opacity $o_i$ and Gaussian distribution of 2DGS, assigning an independent opacity value to each point on the plane, thereby allowing billboards to have **arbitrary shapes** (e.g., leaves, building edges).
    - The **RGB texture $T_i^{\text{RGB}}$** controls per-pixel color offsets; the final color is the sum of the texture sample and the view-dependent color from spherical harmonics: $c_i[u] = T_i^{\text{RGB}}[u] + \text{RGB}(\text{SH}_i, d_i)$.
    - This additive decomposition allows the texture to store only the residual offset relative to the SH base color; most values are near zero, facilitating compression.
    - Values are retrieved from textures via **bilinear sampling**, implemented with PyTorch CUDA kernels that support gradient back-propagation to both the texture and billboard positions.
    - Rendering pipeline: the explicit ray-splat intersection algorithm from 2DGS is used to locate the $(u, v)$ intersection coordinates of a ray with the plane, which are then scaled to $[0, S_T]$ for texture sampling.

2. **Visibility-Weighted Texture Regularization**:

    - Problem: the large number of texture parameters makes the model prone to overfitting to training views.
    - An **influence weight $w_i$** is computed for each billboard based on its alpha-blending contribution $I_i$ over the rendered image (sum of alpha-blended values across all covered pixels), with threshold $\sigma = 500$.
    - $w_i = (\sigma - \min(I_i, \sigma)) / \sigma$ when $I_i > 0$; otherwise $w_i = 0$.
    - L1 regularization pushes the RGB texture toward zero (encouraging sparsity), while the alpha map is pushed toward a Gaussian distribution $\mathcal{G}$.
    - Invisible billboards ($I_i \leq 0$) are exempt from regularization to avoid over-regularizing underrepresented regions.
    - Billboards with large contributions receive small weights (more texture freedom); those with small contributions receive large weights (stronger regularization).

3. **Improved MCMC Density Control**:

    - The 3DGS-MCMC sampling strategy is adopted, but the cloning operation is modified: the original MCMC adjusts opacity and covariance to preserve the rendering state, but covariance adjustment is incompatible with arbitrary-shape primitives.
    - Instead, only the alpha map is adjusted to maintain ray-direction opacity consistency: $T^{\alpha}_{1,\ldots,N} = 1 - (1 - T^{\alpha})^{1/N}$.
    - This formulation is simpler than the original MCMC and requires no additional scale modification.
    - "Dead" primitive criterion: the alpha map mean $\bar{T}_i^{\alpha} < \gamma$ ($\gamma = 5\times10^{-3}$), rather than the global opacity threshold used in the original MCMC.

### Loss & Training

- **Photometric loss**: $\mathcal{L}_{\text{image}} = (1 - \lambda_{\text{SSIM}}) \cdot \mathcal{L}_1 + \lambda_{\text{SSIM}} \cdot \mathcal{L}_{\text{SSIM}}$ ($\lambda_{\text{SSIM}} = 0.2$)
- **Texture regularization**: $\mathcal{L}_{\text{texture}} = \lambda_{\text{RGB}} \cdot \mathcal{L}_{\text{RGB}} + \lambda_{\alpha} \cdot \mathcal{L}_{\alpha}$ ($\lambda_{\text{RGB}} = \lambda_{\alpha} = 1\times10^{-4}$)
- **Total loss**: $\mathcal{L} = \mathcal{L}_{\text{image}} + \mathcal{L}_{\text{texture}}$
- Training runs for 30,000 steps plus an additional 2,000 fine-tuning steps for spherical harmonic coefficients to adapt to the textures.
- The first 500 steps freeze the textures ($T^{\text{RGB}}$ initialized to 0, $T^{\alpha}$ initialized to a 2D Gaussian) to allow SfM initialization to adjust orientations and colors.
- MCMC density control is applied from steps 500 to 25,000.
- Learning rates: $\text{lr}_{\text{RGB}} = 2.5\times10^{-3}$, $\text{lr}_{\alpha} = 1\times10^{-3}$, $\text{lr}_{\text{SH}} = 5\times10^{-3}$, $\text{lr}_{\mu}$ decays exponentially from $1.6\times10^{-4}$ to $1.6\times10^{-6}$.
- For outdoor scenes, 10K uniformly distributed points sampled via the Fibonacci algorithm on a bounding sphere can optionally be added to represent the sky.
- **Texture compression**: normalize to $[0, 1]$ → quantize to uint8 (stored as $\div 4$) → subtract Gaussian template from the alpha map to increase sparsity → ZIP dictionary compression, yielding an average total compression ratio of approximately ×7.
- Hardware: single NVIDIA RTX 4090.

## Key Experimental Results

### DTU Dataset (Comparison with Compression Methods, Table 3)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Storage↓ |
|--------|-------|-------|--------|----------|
| 3DGS | 28.39 | 0.8775 | 0.2850 | 121 MB |
| Reduced-3DGS | 28.31 | 0.8769 | 0.2859 | 30 MB |
| Compact-3DGS | 28.39 | 0.8752 | 0.3000 | 10 MB |
| **BBSplat** | **29.39** | **0.8830** | **0.2770** | **7 MB** |

→ Storage reduced by **×17**, PSNR improved by **+1.0 dB**.

### DTU Dataset (Mesh Extraction + NVS, Table 4, Full HD 1600×1200)

| Method | Chamfer Distance↓ | PSNR↑ | Storage↓ |
|--------|-------------------|-------|----------|
| 3DGS | 1.96 | 28.39 | 121 MB |
| SuGaR | 1.33 | 27.95 | 1059 MB |
| 2DGS | **0.80** | 28.04 | 59 MB |
| **BBSplat** | 0.91 | **29.72** | **32 MB** |

→ PSNR reaches **29.72 (SOTA)**; CD is only 0.11 worse than 2DGS yet substantially better than 3DGS (1.96) and SuGaR (1.33); storage is only 32 MB.

### Tanks & Temples (Fixed Storage Budget, Table 1)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Storage | Primitives |
|--------|-------|-------|--------|---------|------------|
| 3DGS | 20.90 | 0.7810 | 0.3130 | 19 MB | 580K |
| 2DGS† | ~20.5 | ~0.77 | ~0.33 | ~18 MB | ~550K |
| **BBSplat** | **~21.8** | **~0.83** | **~0.22** | **19 MB** | **~60K** |

→ Under the same storage budget, the number of primitives is reduced by approximately one order of magnitude, with all metrics improved.

### Ablation Study (Tanks & Temples, 800×600, Table 5)

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Storage |
|---------------|-------|-------|--------|---------|
| w/o alpha texture $T^{\alpha}$ | 25.17 | 0.8684 | 0.1774 | 60 MB |
| w/o RGB texture $T^{\text{RGB}}$ | 25.26 | 0.8710 | 0.1858 | 72 MB |
| w/o regularization | 25.54 | 0.8791 | 0.1655 | 131 MB |
| w/o compression | 25.71 | 0.8818 | 0.1630 | 662 MB |
| **Full** | **25.68** | 0.8797 | 0.1649 | **93 MB** |

- Removing $T^{\alpha}$ has the largest impact on LPIPS ($0.1649 \to 0.1774$), demonstrating that arbitrary shape is critical for perceptual quality.
- Removing $T^{\text{RGB}}$ leads to noticeable drops in PSNR/SSIM ($-0.42$/$-0.009$).
- Regularization incurs almost no quality loss while significantly reducing storage ($131 \to 93$ MB).
- Compression causes only a marginal quality drop ($25.71 \to 25.68$ PSNR) while reducing storage from 662 MB to 93 MB (×7).

### Texture Size Selection (Table 7, Truck Scene)

| Texture Size | Primitives | PSNR | Storage |
|--------------|------------|------|---------|
| 8×8 | 160K | 26.04 | 207 MB |
| **16×16** | **40K** | **25.99** | **102 MB** |
| 32×32 | 10K | 25.28 | 82 MB |
| 64×64 | 2.5K | 23.34 | 68 MB |

→ $16 \times 16$ achieves the best balance between quality and storage.

## Highlights & Insights

1. **Elegant design**: the texture serves as an offset to the SH color, simultaneously preserving view-dependent effects and promoting texture sparsity for compression.
2. **Plug-and-play**: BBSplat primitives can directly replace Gaussian primitives in any GS pipeline, reusing the ray-splat intersection algorithm from 2DGS.
3. **Three-in-one**: simultaneously addresses rendering quality, mesh extraction accuracy, and storage compression.
4. **Ray-tracing support**: the planar nature of billboards allows direct rasterization in Blender with ray-tracing effects (specular reflection, relighting) without additional mesh extraction steps.
5. **Sophisticated regularization design**: visibility-weighted regularization preserves texture freedom for high-contribution billboards while pushing low-contribution ones toward a Gaussian distribution, simultaneously preventing overfitting and promoting compression.
6. **Revival of a classical CG idea**: Billboard Cloud (2003) is reinvigorated in the era of differentiable rendering.

## Limitations & Future Work

1. **Suboptimal performance on large outdoor scenes**: outdoor scenes in Mip-NeRF-360 containing abundant small monochromatic details (e.g., foliage) are less efficiently represented by textured primitives than by Gaussians.
2. **Slow training**: approximately 40 minutes versus ~5 minutes for 3DGS (×8 slower), with the bottleneck being the backward pass of the CUDA bilinear texture sampling.
3. **Overfitting with large numbers of primitives**: using more primitives in outdoor scenes can paradoxically cause overfitting.
4. **Fixed texture size of $16 \times 16$**: all primitives share the same resolution, with no adaptive adjustment based on primitive importance or size.
5. Comparison with Texture-GS is limited to DTU (Texture-GS cannot handle large scenes due to its spherical texture space definition).

## Related Work & Insights

| Method | Arbitrary Shape | RGB Texture | Storage Compression | Base Framework |
|--------|----------------|-------------|---------------------|----------------|
| 3D Convex Splatting | ✅ | ❌ | ❌ | 3DGS |
| Texture-GS | ❌ | ✅ | ❌ | 3DGS |
| Textured Gaussians | ✅ | ✅ | ✅ | 3DGS |
| SuperGaussians | ✅ | ❌ | ❌ | 2DGS |
| HDGS | ❌ | ✅ | ❌ | 2DGS |
| GSTex | ❌ | ✅ | ❌ | 2DGS |
| Gaussian Billboards | ❌ | ✅ | ❌ | 2DGS |
| **BBSplat** | **✅** | **✅** | **✅** | **2DGS** |

BBSplat is the only 2DGS-based method that simultaneously supports arbitrary shape, RGB textures, and storage compression. Compared to Texture-GS on DTU, BBSplat achieves higher PSNR (30.81 vs. 30.03), and is capable of handling large-scale scenes.

- **Complementarity with 3DGS compression**: BBSplat reduces the number of primitives at the representation level and can be combined orthogonally with parameter compression methods such as Compact-3DGS.
- **The texture-as-offset paradigm** can be generalized to dynamic scenes—time-dependent texture offsets within 4D Gaussians could represent dynamic appearance changes.
- **Visibility-weighted regularization** is broadly applicable and can be adopted in other primitive-based representations requiring overfitting prevention.
- The explicit planar nature of billboards naturally supports editing (translation, deletion, texture modification) and physical effects (reflection, refraction), opening pathways for interactive scene editing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of textured billboards originates from classical CG (2003), but integrating it into a differentiable rendering pipeline with a complete regularization and compression scheme is novel; not a ground-up innovation, but the combination is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three standard datasets with 27 scenes, fixed/maximum storage settings, comparisons with compression methods, mesh extraction evaluation, ablation studies, per-scene metrics, and texture size search; however, an explicit training time comparison table is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, complete mathematical derivations, intuitive figures, and thorough supplementary material (CUDA pseudocode, per-scene metrics, hyperparameter search, training process visualization).
- **Value**: ⭐⭐⭐⭐ Practically addresses two important problems—insufficient quality of 2DGS and excessive storage of GS; ray-tracing support and plug-and-play capability have practical utility; the ×8 training slowdown is the primary bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](../../CVPR2025/3d_vision/novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[ICCV 2025\] HumanOLAT: A Large-Scale Dataset for Full-Body Human Relighting and Novel-View Synthesis](humanolat_a_large-scale_dataset_for_full-body_human_relighting_and_novel-view_sy.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)

</div>

<!-- RELATED:END -->
