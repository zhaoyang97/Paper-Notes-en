---
title: >-
  [Paper Note] RadioGS: Radiometrically Consistent Gaussian Surfels for Inverse Rendering
description: >-
  [ICLR 2026][3D Vision][Inverse Rendering] RadioGS introduces a radiometric consistency loss that minimizes the residual between the learned radiance of each Gaussian surfel and its physically rendered radiance…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Inverse Rendering"
  - "Gaussian Splatting"
  - "Indirect Illumination"
  - "Radiometric Consistency"
  - "Ray Tracing"
date: 2026-05-08
content_hash: c70f2f1d5fa86d70
---

# RadioGS: Radiometrically Consistent Gaussian Surfels for Inverse Rendering

**Conference**: ICLR 2026
**arXiv**: [2603.01491](https://arxiv.org/abs/2603.01491)
**Code**: [https://qbhan.github.io/radiogs-page/](https://qbhan.github.io/radiogs-page/)
**Area**: 3D Vision
**Keywords**: Inverse Rendering, Gaussian Splatting, Indirect Illumination, Radiometric Consistency, Ray Tracing

## TL;DR

RadioGS introduces a radiometric consistency loss that minimizes the residual between the learned radiance of each Gaussian surfel and its physically rendered radiance, providing physics-based supervision for unobserved directions. This forms a self-correcting feedback loop that enables accurate indirect illumination and material decomposition, while supporting relighting in minutes.

## Background & Motivation

**Background**: Inverse rendering based on Gaussian Splatting has advanced rapidly, efficiently recovering geometry, materials, and lighting from multi-view images. However, accurately decomposing global illumination effects—especially indirect illumination and inter-surface reflections—remains a core challenge.

**Limitations of Prior Work**: Existing methods handle indirect illumination in two main ways: (1) treating indirect radiance as a learnable residual (e.g., R3DG, GS-IR), where unconstrained optimization leads to ambiguous decomposition of lighting and materials; (2) querying indirect radiance from pre-trained NVS Gaussian primitives (e.g., IRGS, SVG-IR), where pre-training is supervised only for training viewpoints, making radiance queried from unobserved directions potentially incorrect.

**Key Challenge**: NVS training constrains the radiance of Gaussians only in camera-visible directions, whereas indirect illumination requires querying radiance from arbitrary directions (including inter-surface reflection directions). The lack of supervision for unobserved directions leads to inaccurate indirect radiance, causing lighting to be incorrectly baked into surface materials.

**Goal**: To provide a physics-based constraint that enables Gaussian surfels to obtain correct radiance values even in unobserved directions, thereby accurately modeling indirect illumination and inter-surface reflections.

**Key Insight**: Drawing inspiration from self-training radiance caches—iteratively minimizing the rendering equation residual to drive Gaussian primitive radiance toward physically correct solutions.

**Core Idea**: Radiometric consistency enforces that the learned radiance $L_\mathbf{G}$ of each Gaussian surfel agrees with its physically rendered radiance $L_\mathbf{G}^{PBR}$ derived from the rendering equation, forming a self-correcting loop: reconstruction supervision from camera viewpoints propagates to indirect illumination terms, while physical rendering in turn constrains radiance in unobserved directions.

## Method

### Overall Architecture

The pipeline consists of two stages. In the **initialization stage**, a simplified radiometric consistency loss with split-sum approximation is combined with an NVS reconstruction loss to pre-train Gaussian surfels and establish stable geometry. In the **inverse rendering stage**, the full Monte Carlo radiometric consistency loss is jointly optimized with material smoothness and lighting prior losses to refine geometry, materials, and lighting. For **relighting**, geometry and materials are fixed, and only surfel radiance is fine-tuned under the new lighting (~2 minutes), after which rendering proceeds directly using surfel radiance (<10ms/frame).

### Key Designs

1. **Radiometric Consistency Loss**:

    - **Function**: Provides physics-based supervision for surfel radiance in unobserved directions.
    - **Mechanism**: For each surfel at position $x$ and outgoing direction $\omega_o$, the physically rendered radiance is computed as $L_\mathbf{G}^{PBR}(x,\omega_o) = \int f_r \cdot (V \cdot L_{dir} + L_{ind}) \cdot (\omega_i \cdot n_x) d\omega_i$, where visibility $V$ and indirect radiance $L_{ind}$ are obtained via 2D Gaussian ray tracing. The residual $\mathcal{R}_\mathbf{G} = L_\mathbf{G} - L_\mathbf{G}^{PBR}$ defines the loss $\mathcal{L}_{rad} = \mathbb{E}_{j,\omega_o}[\|\mathcal{R}_\mathbf{G}\|_1]$. Minimizing this residual creates a bidirectional feedback: physical rendering guides radiance in unobserved directions, while camera-constrained radiance propagates through indirect illumination terms to other surfels.
    - **Design Motivation**: The self-correcting loop relies on the fact that reconstruction losses at camera viewpoints ensure accuracy for a subset of directions; this accurate radiance propagates via ray tracing as indirect illumination to other surfels, which in turn constrains their radiance values.

2. **2D Gaussian Ray Tracing and Monte Carlo Sampling**:

    - **Function**: Efficiently and differentiably obtains inter-surfel visibility and indirect radiance.
    - **Mechanism**: A 2D Gaussian ray tracer casts rays as $\text{Trace}(x, \omega_i; \mathbf{G}) = (L_{trace}, T_{trace})$, where accumulated radiance $L_{trace}$ serves as indirect radiance $L_{ind}$ and $1-T_{trace}$ serves as visibility $V$. Monte Carlo estimation randomly samples $N_g = 4096$ surfels per step, each uniformly sampling $N_s = 64$ incident rays over its normal-defined hemisphere (totaling $2^{18}$ rays), while also sampling random outgoing directions (unobserved) and camera directions (constrained).
    - **Design Motivation**: The 2D Gaussian ray tracer shares ray-splat intersection computations with Gaussian surfels, enabling seamless and differentiable integration. Sampling camera directions ensures that constrained radiance signals propagate to ray-traced surfels.

3. **Fine-Tuning-Based Efficient Relighting**:

    - **Function**: Rapidly adapts surfel radiance to new lighting conditions.
    - **Mechanism**: Given new lighting, only a few rounds of fine-tuning via minimization of $\mathcal{L}_{rad}$ are required (~2 minutes). After fine-tuning, rendering from arbitrary viewpoints using surfel radiance directly is possible (<10ms/frame), without runtime ray tracing or per-surfel storage of multi-directional incident radiance.
    - **Design Motivation**: Traditional methods require runtime indirect radiance queries during relighting (expensive), whereas radiometric consistency fine-tuning allows surfels to directly "memorize" the correct radiance under new lighting.

### Loss & Training

Initialization stage: $\mathcal{L}_{init} = \mathcal{L}_{recon} + \mathcal{L}_{recon}^{PBR} + \lambda_{rad}\mathcal{L}_{rad} + \lambda_{dist}\mathcal{L}_{dist} + \lambda_n\mathcal{L}_n + \lambda_{ns}\mathcal{L}_{ns} + \lambda_m\mathcal{L}_m$ (split-sum approximation of radiometric consistency). Inverse rendering stage: $\mathcal{L}_{inv} = \mathcal{L}_{init} + \lambda_{as}\mathcal{L}_{as} + \lambda_{rs}\mathcal{L}_{rs} + \lambda_{light}\mathcal{L}_{light}$ (full Monte Carlo radiometric consistency + material smoothness + lighting prior). The radiometric consistency weight is $\lambda_{rad} = 0.2$ (inverse rendering) and $1.0$ (relighting fine-tuning). Total training time is approximately 60 minutes on an RTX 4090.

## Key Experimental Results

### Main Results

| Method | NVS PSNR↑ | Normal MAE↓ | Albedo PSNR↑ | Relight PSNR↑ | Training Time |
|--------|-----------|-------------|--------------|---------------|---------------|
| TensoIR (NeRF) | 35.09 | 4.10 | 29.27 | 28.58 | 4h |
| GS-IR | 35.33 | 4.95 | 29.94 | 24.37 | - |
| IRGS | - | - | - | - | - |
| SVG-IR | - | - | - | - | - |
| **RadioGS** | **Best** | **Best** | **Best** | **Best** | **1h** |

RadioGS outperforms existing GS-based and NeRF-based methods on nearly all metrics on the TensoIR dataset while maintaining computational efficiency.

### Ablation Study

| Configuration | Relight PSNR | Notes |
|---------------|-------------|-------|
| Full RadioGS | Best | Full Monte Carlo radiometric consistency |
| w/o radiometric consistency loss | Significant drop | Inaccurate indirect illumination |
| Split-sum only (no MC) | Drop | Approximation insufficient for complex reflections |
| w/o initialization-stage consistency | Drop | Unstable geometric foundation |
| Fine-tuning relight vs. RT relight | Slightly lower but much faster | <10ms vs. ~100ms |

### Key Findings

- The radiometric consistency loss is the central source of RadioGS's advantage—removing it substantially degrades relighting quality, confirming the necessity of physics-based constraints for indirect illumination modeling.
- The inter-surface reflection of a red light bulb on a yellow LEGO surface (TensoIR dataset) demonstrates RadioGS's precise modeling of inter-surface reflections, which competing methods tend to bake into albedo.
- The fine-tuning relighting strategy achieves quality close to ray-tracing-based relighting with only 2 minutes of training, while rendering an order of magnitude faster.

## Highlights & Insights

- The **self-correcting feedback loop** design is particularly insightful—NVS supervision and physical constraints are complementary rather than competing: NVS constrains observed directions while physical rendering constrains unobserved directions, with the two connected through indirect illumination to form a closed loop.
- The **simplified radiometric consistency in the initialization stage** represents an important engineering insight—applying Monte Carlo sampling directly on unstable geometry causes training oscillations, and the split-sum approximation provides a smooth transition.
- **Fine-tuning-based relighting** converts the cost of runtime ray tracing into an offline fine-tuning cost, making it well-suited for applications requiring multi-frame rendering.

## Limitations & Future Work

- The method assumes dielectric materials; its effectiveness on strongly specular materials such as metals has not been validated.
- The $2^{18}$ ray traces per step still incur computational overhead, with training taking approximately one hour.
- Monte Carlo sampling may produce inaccurate estimates in regions with low surfel density.
- Fine-tuning relighting may require more iterations when lighting changes drastically (e.g., from indoor to outdoor settings).

## Related Work & Insights

- **vs. IRGS (Gu et al., 2024)**: IRGS also uses Gaussian ray tracing to optimize indirect radiance, but training signals still come exclusively from observed viewpoint images. RadioGS provides additional supervision for unobserved directions through physics-based constraints.
- **vs. SVG-IR (Sun et al., 2025)**: SVG-IR queries indirect radiance from NVS pre-trained Gaussian point clouds, but pre-trained Gaussians are unconstrained in unobserved directions. RadioGS's radiometric consistency addresses this fundamental issue.
- **vs. Neural Radiance Cache (Müller et al., 2021)**: Radiance caching is used for global illumination in forward rendering; RadioGS extends this concept to Gaussian primitives in inverse rendering.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The self-correcting feedback loop via radiometric consistency loss is novel and physically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on both synthetic and real datasets with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation, method derivation, and experimental analysis are all exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Significant advancement in accurate indirect illumination modeling for GS-based inverse rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering](../../CVPR2026/3d_vision/sgs-intrinsic_semantic-invariant_gaussian_splatting_for_sparse-view_indoor_invers.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](../../ICCV2025/3d_vision/geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[ICLR 2026\] 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)
- [\[ICLR 2026\] Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer](color3d_controllable_and_consistent_3d_colorization_with_personalized_colorizer.md)
- [\[ICLR 2026\] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image](one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)

</div>

<!-- RELATED:END -->
