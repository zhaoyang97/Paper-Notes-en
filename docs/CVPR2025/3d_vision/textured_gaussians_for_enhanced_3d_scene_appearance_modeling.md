---
title: >-
  [Paper Note] Textured Gaussians for Enhanced 3D Scene Appearance Modeling
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Textured Gaussians introduces traditional graphics texture mapping and alpha mapping into 3DGS. By assigning an independent 2D RGBA texture map to each Gaussian, a single Gaussian can represent spatially varying color and opacity. This significantly enhances the representation capability of 3DGS, improving rendering quality given the same budget of Gaussians and yielding an almost 2dB PSNR gain at 1% Gaussian count.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Texture Mapping"
  - "Novel View Synthesis"
  - "Appearance Modeling"
  - "Alpha Mapping"
date: 2026-05-08
content_hash: 61b8125e771d6866
---

# Textured Gaussians for Enhanced 3D Scene Appearance Modeling

**Conference**: CVPR 2025  
**arXiv**: [2411.18625](https://arxiv.org/abs/2411.18625)  
**Code**: [https://textured-gaussians.github.io](https://textured-gaussians.github.io)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Texture Mapping, Novel View Synthesis, Appearance Modeling, Alpha Mapping

## TL;DR
Textured Gaussians introduces traditional graphics texture mapping and alpha mapping into 3DGS. By assigning an independent 2D RGBA texture map to each Gaussian, a single Gaussian can represent spatially varying color and opacity. This significantly enhances the representation capability of 3DGS, improving rendering quality given the same budget of Gaussians and yielding an almost 2dB PSNR gain at 1% Gaussian count.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the state-of-the-art method for novel view synthesis. With its advantages of high-quality rendering, fast training and inference, and explicit representation, it is widely applied in reconstruction, scene editing, and human body modeling tasks.

**Limitations of Prior Work**: A single Gaussian in 3DGS has two fundamental limitations: (1) **Uniform color**—all pixels covered by the same Gaussian are shaded with the same color (scaled only by the Gaussian decay factor), failing to represent spatially varying texture details; (2) **Restricted shape**—each Gaussian can only represent an ellipsoidal shape, making it unable to represent complex geometry. This implies that 3DGS requires a huge number of Gaussians to fit high-frequency textures and fine geometry, leading to high memory and compute overheads.

**Key Challenge**: The trade-off between the expressive power of a single Gaussian primitive and the overall model efficiency. Rich rendering demands dense Gaussians, but having too many Gaussians degenerates training and rendering efficiency.

**Goal**: To significantly enhance the expressive power of a single Gaussian under the premise of keeping the overall 3DGS framework unchanged, allowing it to achieve equivalent or better rendering quality with far fewer Gaussians.

**Key Insight**: The authors draw inspiration from traditional mesh rendering. A mesh surface uses texture mapping to express complex appearances, where each triangle corresponds to a region on a texture map. Analogously, each Gaussian can also have its own texture map.

**Core Idea**: Assign a fixed-resolution 2D texture map (supporting Alpha-only / RGB / RGBA) to each 3D Gaussian, and query the texture values via ray-Gaussian intersection and UV mapping to achieve spatially varying color and opacity.

## Method

### Overall Architecture
Textured Gaussians is built upon 3DGS, with the rendering pipeline: (1) emitting a ray from the camera center toward each pixel; (2) computing the intersection of the ray with the 3D Gaussians in the scene; (3) querying the Gaussian's texture map at the intersection point via UV mapping to obtain the texture color $\mathbf{c}^{tex}$ and texture alpha value $\alpha^{tex}$; (4) compounding the texture color with the SH base color, multiplying the texture alpha with the Gaussian opacity, and performing front-to-back alpha blending to compute the final pixel color.

### Key Designs

1. **Ray-Gaussian Intersection and UV Mapping**:

    - **Function**: Maps pixel rays to local texture coordinates of the Gaussian.
    - **Mechanism**: Each Gaussian is defined by three principal axes. The two axes with the largest scales define a plane $\mathcal{P}$, with the normal direction being the axis with the smallest scale. For pixel $\mathbf{p}$, a ray is emitted from the camera origin $\mathbf{o}$ and intersected with plane $\mathcal{P}$ to get a 3D intersection point $\mathbf{x}$. The intersection point is then projected onto the two principal axes of the Gaussian and normalized to the texture coordinates $(u, v)$ in $[0, \mathcal{T}-1]$: $u = \frac{m \cdot \sigma_1 + (\mathbf{x} - \boldsymbol{\mu}) \cdot \mathbf{r}_1}{2m \cdot \sigma_1} \cdot (\mathcal{T}-1)$, where $\sigma_1, \sigma_2$ are scales, $\mathbf{r}_1, \mathbf{r}_2$ are axis directions, and $m=3$ is the range multiplier.
    - **Design Motivation**: This UV mapping approach naturally wraps the texture onto the "surface" of the Gaussian and is fully compatible with the geometric transformation (rotation, scaling) of the Gaussian.

2. **Generalized Gaussian Appearance Model**:

    - **Function**: Unifies the rendering equations of original 3DGS and Textured Gaussians.
    - **Mechanism**: The rendering equation of the final pixel color is extended to: $\mathbf{c}_{final}(\mathbf{p}) = \sum_{i=1}^K \mathbf{c}_i(\mathbf{p}) \cdot \alpha_i(\mathbf{p}) \cdot \prod_{j=1}^{i-1}(1-\alpha_j(\mathbf{p}))$, where color is $\mathbf{c}_i(\mathbf{p}) = \mathbf{c}_i^{base} + \mathbf{c}_i^{tex}(u,v)$ (SH base color + texture color) and alpha is $\alpha_i(\mathbf{p}) = \alpha_i^{tex}(u,v) \cdot \mathcal{G}_i(\mathbf{x}) \cdot o_i$ (texture alpha $\times$ Gaussian value $\times$ opacity). When $\mathbf{c}^{tex}=0, \alpha^{tex}=1$, it degenerates to original 3DGS.
    - **Design Motivation**: RGB textures enable Gaussians to represent high-frequency color variations; alpha textures allow Gaussians to break free from ellipsoidal boundary restrictions—spatially varying transparency can "carve" out arbitrary shapes.

3. **Two-Stage Optimization Strategy**:

    - **Function**: Stabilizes training and avoids ill-posed problems of joint optimization.
    - **Mechanism**: **Stage 1** (30K iterations): Optimize all Gaussian properties (position, rotation, scale, SH coefficients, opacity) under the standard 3DGS pipeline, including Adaptive Density Control (ADC). **Stage 2** (30K iterations): Initialize all Gaussian properties with Stage 1 results, freeze ADC, and jointly optimize Gaussian properties and texture map parameters. Texture RGB is initialized with near-zero values ($25/255$), and the alpha channel is initialized with 1.
    - **Design Motivation**: Jointly optimizing all parameters is a highly ill-posed problem (there is massive ambiguity between Gaussian positions and texture contents). This two-stage strategy allows Stage 1 to first establish the geometric layout before Stage 2 learns the texture details.

### Loss & Training
The standard weighted photometric loss is used: $\mathcal{L} = \lambda \mathcal{L}_1 + (1-\lambda) \mathcal{L}_{SSIM}$ with $\lambda=0.8$. The learning rate for texture maps is set to 0.001, which is applied uniformly across all datasets without custom tuning. All experiments are conducted on an NVIDIA H100 GPU cluster.

## Key Experimental Results

### Main Results

Evaluate PSNR/SSIM/LPIPS on 5 standard benchmark datasets, comparing with 3DGS* (an improved 3DGS implementation) under the same number of Gaussians.

| Dataset | 3DGS* (PSNR/SSIM/LPIPS) | RGBA Textured (PSNR/SSIM/LPIPS) | 1% GS 3DGS* | 1% GS Ours |
|--------|--------------------------|----------------------------------|-------------|------------|
| Blender | 33.09/0.967/0.044 | **33.31/0.969/0.038** | 26.89/0.916/0.117 | **28.02/0.934/0.085** |
| Mip-NeRF 360 | 27.28/0.832/0.187 | **27.43/0.838/0.176** | 22.37/0.629/0.477 | **23.75/0.707/0.337** |
| DTU | 33.54/0.970/0.055 | **33.68/0.972/0.050** | 30.88/0.932/0.158 | **32.41/0.963/0.070** |
| Tanks & Temples | 24.18/0.854/0.175 | **24.39/0.860/0.163** | 19.90/0.674/0.441 | **21.08/0.738/0.311** |
| Deep Blending | 28.04/0.894/0.271 | **28.52/0.902/0.239** | 23.97/0.817/0.434 | **24.88/0.845/0.371** |

### Ablation Study (Texture Type Ablation)

| Texture Type | Blender (PSNR) | Mip-NeRF 360 | DTU | T&T | DB|
|----------|---------------|-------------|-----|-----|-----|
| 3DGS* (No Texture) | 33.09 | 27.28 | 33.54 | 24.18 | 28.04 |
| Alpha-only | **33.22** | 27.32 | 33.51 | 24.27 | **28.36** |
| RGB-only | 33.18 | 27.30 | 33.48 | 24.22 | 28.30 |
| RGBA | **33.31** | **27.43** | **33.68** | **24.39** | **28.52** |

### Key Findings
- **The performance of Alpha-only textures is surprisingly strong**: using only the alpha channel (1/4 the model size of RGBA) outperforms RGB-only textures and closely approaches RGBA. This is because alpha textures allow Gaussians to break free from ellipsoidal constraints and achieve complex shapes through spatially varying opacity, whereas RGB-only textures remain confined to ellipsoidal shapes.
- The smaller the number of Gaussians, the larger the advantage of Textured Gaussians: producing nearly a 2dB PSNR gain at 1% Gaussian count, and a 0.2-0.5dB improvement at 100% Gaussian budget.
- Under the same model size budget (by reducing the number of Gaussians to compensate for texture overhead), alpha-only textures usually perform the best, indicating an optimal allocation ratio between Gaussian structural parameters and texture parameters.
- A sweet spot exists between texture resolution and the number of Gaussians: with a fixed budget, the best performance is reached neither at maximum texture resolution nor at the highest Gaussian count, but at a balance between the two.

## Highlights & Insights
- **The finding that alpha textures are more critical than RGB textures is counter-intuitive yet profound**: It reveals that the bottleneck of 3DGS lies in shape representation rather than color representation. The constraint of ellipsoidal shapes degrades rendering quality more severely than the constraint of single colors. This insight can guide future directions for modifying 3DGS.
- **The combination of texture mapping and Gaussian splatting is highly natural**: Ray-Gaussian intersection is already required, introducing minimal extra computational overhead and maintaining almost identical inference speeds.
- **The two-stage training strategy is generalizable to other 3DGS expansion methods**: Any approach that appends extra attributes to Gaussians can adopt the strategy of 'first optimizing geometry, then optimizing added attributes.'

## Limitations & Future Work
- Currently uses 2D diffuse textures, which cannot model spatially varying specular colors; future work could extend this to 3D volumetric textures or 5D radiance fields.
- The two-stage training nearly doubles the training time compared to standard 3DGS.
- Compared to concurrent work GStex, GStex is based on 2D Gaussian Splatting and lacks alpha channel support, constraining its representation capabilities.
- The texture resolution is fixed. Using the same resolution across different scales of Gaussians can be inflexible—larger Gaussians command higher-resolution textures.

## Related Work & Insights
- **vs 3DGS (Original)**: 3DGS assigns only one color and one ellipsoidal shape per Gaussian. Textured Gaussians significantly improves primitive expressiveness through texture mapping.
- **vs Textured-GS (Huang & Gong)**: This method smoothly varies colors inside a Gaussian by adjusting view direction computation. However, due to the smoothness of SH representations, it cannot reconstruct high-frequency textures. The texture maps in this paper do not suffer from this limitation.
- **vs Texture-GS (Xu et al.)**: This method utilizes a global texture map alongside learned UV mappings. Restricted by spherical parameterizations, it struggles with complex geometries and large-scale scenes. The per-Gaussian independent textures proposed in this work are more flexible.
- **vs GStex (Rong et al., concurrent)**: GStex distributes texels based on 2D Gaussian discs but lacks an alpha channel, limiting its ability to describe complex shapes.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of introducing texture mapping to 3DGS is natural yet effective, and the discovery of alpha textures is a highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablations across 5 datasets plus custom datasets, varying Gaussian counts, texture resolutions, and texture types.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear method formulation, complete equation derivations, and highly systematic ablation experiments.
- Value: ⭐⭐⭐⭐ Enhancing single Gaussian expressiveness is of high practical significance, especially in memory-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[ICCV 2025\] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](../../ICCV2025/3d_vision/can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)
- [\[CVPR 2025\] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](isegman_interactive_segment-and-manipulate_3d_gaussians.md)
- [\[CVPR 2025\] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning](reconstructing_close_human_interaction_with_appearance_and_proxemics_reasoning.md)

</div>

<!-- RELATED:END -->
