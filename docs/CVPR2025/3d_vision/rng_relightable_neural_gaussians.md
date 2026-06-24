---
title: >-
  [Paper Note] RNG: Relightable Neural Gaussians
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This work proposes the Relightable Neural Gaussians (RNG) framework, which learns a latent vector for each Gaussian element conditioned on the view and light directions. By integrating shadow cues and a hybrid forward-deferred optimization strategy, RNG achieves high-quality relighting of soft-boundary objects.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Relighting"
  - "Neural Gaussians"
  - "Shadow Mapping"
  - "Hybrid Rendering"
date: 2026-05-08
content_hash: 27c44af8054530dd
---

# RNG: Relightable Neural Gaussians

**Conference**: CVPR 2025  
**arXiv**: [2409.19702](https://arxiv.org/abs/2409.19702)  
**Code**: [https://whois-jiahui.fun/project_pages/RNG](https://whois-jiahui.fun/project_pages/RNG)  
**Area**: 3D Vision / Relighting  
**Keywords**: 3D Gaussian Splatting, Relighting, Neural Gaussians, Shadow Mapping, Hybrid Rendering

## TL;DR

This work proposes the Relightable Neural Gaussians (RNG) framework, which learns a latent vector for each Gaussian element conditioned on the view and light directions. By integrating shadow cues and a hybrid forward-deferred optimization strategy, RNG achieves high-quality relighting of soft-boundary objects.

## Background & Motivation

Creating relightable 3D assets is crucial for content creation, but decomposing lighting, geometry, and material is inherently an ill-posed problem. Most existing methods rely on analytical shading models and surface constraints (such as valid normal assumptions) and struggle with blurry-boundary objects like fur and fabrics. Although NRHints supports relighting of soft-boundary objects, it is based on NeRF, leading to slow training/rendering and over-smoothing. Concurrent work GS3 utilizes 3DGS for efficiency but suffers from poor shadow quality due to inaccurate geometry. The key challenge is how to achieve precise shadows using the high efficiency of 3DGS while avoiding analytical shading assumptions. This paper addresses this by replacing analytical models with neural latent vectors, improving shadow quality via shadow mapping and a depth refinement network, and balancing geometry and appearance using a hybrid forward-deferred optimization.

## Method

### Overall Architecture

Each Gaussian point carries a latent vector representing its reflectance, which is decoded into a color by a neural decoder $\Theta$ conditioned on the view direction $\omega_o$, lighting direction $\omega_i$, and shadow cue $V$. The training is split into two stages: the first stage uses forward shading (decoding before blending) to obtain geometry and initialize the latent vectors; the second stage uses deferred shading (blending features before decoding) with shadow cues enabled to optimize shadow quality. Inference achieves 60 FPS on an RTX 4090.

### Key Designs

1. **Relightable Neural Gaussians**: Each Gaussian point stores a latent vector (feature vector) instead of spherical harmonics coefficients or analytical BRDF parameters. The reflectance is represented as $\rho(\mathbf{x}, \omega_o, \omega_i) = \Theta(\mathbf{x} | \omega_o, \omega_i, V)$, where $\Theta$ is an MLP decoder. This completely avoids assumptions about the type of shading model (e.g., Disney BRDF) or surface constraints (e.g., valid normals), enabling the framework to learn appearances that do not conform to simple analytical models—particularly suitable for blurry materials like fur and fabrics.

2. **Shadow Cues + Depth Refinement**: Point light sources generate sharp shadows, whereas MLPs tend to over-smooth. Shadow cues are acquired by performing shadow mapping within the 3DGS framework: cameras are first used to obtain the camera depth through splatting, which is then corrected by a depth refinement network $\bar{z}' = \bar{z} \cdot \Phi(\omega_o)$ (since the weighted depth sum might be inaccurate) to locate the shading point $P$. Then, a virtual camera splats from the light source position to obtain the shadow depth and find the intersection point $Q$, recording $|PQ|$ as the shadow cue. This cue serves as an additional input to the decoder, significantly improving shadow clarity and consistency.

3. **Hybrid Forward-Deferred Optimization**: Forward shading $C_{\text{forward}} = \sum \Theta(\mathbf{x}_i | \omega_o, \omega_i) \alpha_i \prod(1-\alpha_j)$ decodes before blending, which yields good geometry but blurry shadows (blending blurs high frequencies). Deferred shading $C_{\text{defer}} = \Theta(\sum \mathbf{x}_i \alpha_i \prod(1-\alpha_j) | \omega_o, \omega_i, V)$ blends features before decoding, resulting in sharp shadows but potential floaters. A two-stage strategy is adopted: the first stage uses forward shading to optimize geometry and latent vectors, while the second stage applies deferred shading combined with shadow cues to refine the appearance.

### Loss & Training

- L1 + SSIM image reconstruction loss
- Training takes approximately 1.3 hours (RTX 4090)
- Shadow cues are disabled in the first stage (imperfect initial Gaussian shapes could provide wrong shadow information and disrupt training)
- Latent vectors from the first stage serve as initialization for the second stage to accelerate convergence
- Inputs are multi-view images under a moving point light source

## Key Experimental Results

### Main Results

| Method | Framework | PSNR↑ | SSIM↑ | LPIPS↓ | Training Time | Rendering FPS |
|------|------|-------|-------|--------|---------|---------|
| NRHints | NeRF | 27.38 | 0.860 | 0.133 | ~24h | ~1 |
| GS3 | 3DGS | Comparable | Comparable | Comparable | ~1.5h | ~60 |
| **RNG** | **3DGS** | **Best/Second-best** | **Best/Second-best** | **Best/Second-best** | **~1.3h** | **~60** |

### Ablation Study

| Component | PSNR Change | Shadow Quality |
|------|----------|---------|
| W/o shadow cues | Decrease | Poor (blurry/inconsistent) |
| W/o depth refinement | Slight decrease | Moderate (shifted shadows) |
| Forward shading only | Comparable | Poor (blurry shadows) |
| Deferred shading only | Decrease | Good but with floaters |
| **Full RNG** | **Highest** | **Best** |

### Key Findings

- RNG achieves the best or second-best metrics across most scenarios, producing the overall best average PSNR, SSIM, and LPIPS.
- Training is approximately 18 times faster than NRHints, and the rendering speed is roughly 60 times faster.
- Shadow quality is significantly superior to GS3, due to the shadow cues and depth refinement.
- The hybrid optimization strategy effectively balances geometry quality and shadow sharpness.
- The relighting results for soft-boundary objects (e.g., hair, fabrics) are notably better than those from methods relying on surface constraints.

## Highlights & Insights

- Replacing analytical shading models with latent vectors is a sound design choice, as real-world materials rarely fit into simple analytical models.
- The formulation of shadow mapping within the 3DGS framework is novel, simulating ray tracing using two splatting passes (camera and light source viewpoints).
- The depth refinement network successfully addresses the practical issue of inaccurate weighted Gaussian depths.
- The hybrid forward-deferred design is backed by strong physical intuition: the position of the blending operation determines its capability for frequency retention.

## Limitations & Future Work

- Currently, only single point-light relighting is supported, and environmental lighting incurs high integration overhead.
- The depth refinement assumes a linear correction, which might be insufficient for highly complex geometries.
- The resolution of the shadow maps may limit extremely fine shadow details.
- Future work could extend this to multi-light sources, environment maps, and dynamic scenes.

## Related Work & Insights

- **vs NRHints**: Both methods support soft boundaries, but NRHints is NeRF-based and over-smooths elements. RNG uses 3DGS to retain more details while being 18 times faster.
- **vs GS3**: Both are based on 3DGS, but GS3 relies on analytical approximations. RNG achieves superior shadow quality via its shadow cues and depth refinement.
- **vs 3DGS Inverse Rendering Methods**: Methods like GaussianShader rely on surface normals and analytical BRDFs, making them unsuitable for soft-boundary objects.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of neural Gaussians, shadow mapping, and hybrid optimization is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Thorough evaluation with multi-scene comparisons and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-explained methodology with well-founded design choices.
- **Value**: ⭐⭐⭐⭐ — Highly practical with 1.3 hours of training and 60 FPS rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[NeurIPS 2025\] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](../../NeurIPS2025/3d_vision/becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)
- [\[CVPR 2025\] ARM: Appearance Reconstruction Model for Relightable 3D Generation](arm_appearance_reconstruction_model_for_relightable_3d_generation.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](../../ICCV2025/3d_vision/gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)
- [\[CVPR 2025\] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](isegman_interactive_segment-and-manipulate_3d_gaussians.md)

</div>

<!-- RELATED:END -->
