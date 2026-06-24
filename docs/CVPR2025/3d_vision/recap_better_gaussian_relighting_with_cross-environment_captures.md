---
title: >-
  [Paper Note] ReCap: Better Gaussian Relighting with Cross-Environment Captures
description: >-
  [CVPR 2025][3D Vision][3D Gaussian] ReCap leverages multiple sets of images of the same object under different lighting environments as multi-task supervision signals, sharing material properties while independently optimizing lighting representations. This fundamentally resolves the albedo-lighting ambiguity. Combined with simplified shading functions and HDR post-processing, it significantly outperforms all existing methods on an expanded relighting benchmark.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian"
  - "Relighting"
  - "Inverse Rendering"
  - "Cross-environment Captures"
  - "Material-lighting Decoupling"
date: 2026-05-08
content_hash: 5c8e07d5c954cd0b
---

# ReCap: Better Gaussian Relighting with Cross-Environment Captures

**Conference**: CVPR 2025  
**arXiv**: [2412.07534](https://arxiv.org/abs/2412.07534)  
**Code**: [GitHub](https://github.com/TRI-ML/recap)  
**Area**: 3D Vision / Relighting  
**Keywords**: 3D Gaussian, Relighting, Inverse Rendering, Cross-environment Captures, Material-lighting Decoupling

## TL;DR

ReCap leverages multiple sets of images of the same object under different lighting environments as multi-task supervision signals, sharing material properties while independently optimizing lighting representations. This fundamentally resolves the albedo-lighting ambiguity. Combined with simplified shading functions and HDR post-processing, it significantly outperforms all existing methods on an expanded relighting benchmark.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become a mainstream 3D representation due to its high-quality rendering and real-time frame rates. Subsequent works (such as GShader, GS-IR, R3DG, etc.) endow 3DGS with relighting capabilities through explicit shading functions and learnable environment lighting representations.

**Limitations of Prior Work**: Due to the albedo-lighting ambiguity, reconstruction loss alone cannot correctly separate material and lighting—variations in surface albedo and changes in light intensity are indistinguishable in appearance. The environment lightmaps learned by existing methods are often contaminated with object color, hue shifts, intensity scaling, and noise, becoming a "residual dustbin" during optimization. Rendering quality drops significantly when replacing them with real HDR environment maps.

**Key Challenge**: Training data constraint under a single lighting environment is insufficient to uniquely determine the decomposition of material and lighting—the same observation can be explained by an infinite number of material-lighting combinations.

**Goal**: To provide the missing photometric constraints for inverse rendering by introducing multiple sets of appearance data across different lighting environments, thereby breaking the ambiguity.

**Key Insight**: Inspired by photometric appearance modeling, multiple sets of images of the same object under different unknown lighting are leveraged, analogous to multi-task learning—multiple "task heads" (lighting representations) share a "backbone" (material properties), with the shading function acting as a "bridge" for physical constraints.

**Core Idea**: To treat cross-environment capture as a multi-task objective, jointly optimizing shared material properties and independent lighting representations. The PBR shading function ensures both the physical plausibility of the decomposition and cross-environment consistency.

## Method

### Overall Architecture

Based on the 3DGS framework, each Gaussian point is augmented with 3 material attributes (base color $\mathbf{b}$, roughness $r$, specular tint $\mathbf{s}$). Given k sets of images under different lighting environments, k learnable $6\times256\times256$ cube environment lightmaps are instantiated. During rendering, the corresponding environment lightmap is queried based on the normal direction, the Gaussian color is calculated via the shading function, and the loss is computed after standard splatting rasterization. All k sets of lighting representations share a single, unique material model.

### Key Designs

1. **Disambiguated Split Sum Approximation**:

    - **Function**: Simplifies the Disney Principled BRDF shading function and eliminates the optimization ambiguity caused by the metallic parameter.
    - **Mechanism**: The original split-sum linearly interpolates between metallic and non-metallic models using a metallic parameter $m$. The authors find that when $E_d \sim E_s\beta_1$, the two models become interchangeable, making the optimization of $m$ unstable. By removing $m$ and expanding the scalar specular to a 3-channel vector $\mathbf{s} \in [0,1]^3$, they obtain the general formula $L_{\text{out}} = E_s \mathbf{s}\beta_1 + E_s\beta_2 + E_d \mathbf{b}$. Saturation penalty $\mathcal{L}_{\text{sat}}$ and energy conservation constraint $\|\mathbf{s}\| + \|\mathbf{b}\| \leq 1$ are applied to avoid non-physical parameters.
    - **Design Motivation**: The duality of metallic (metal vs. non-metal) introduces optimization ambiguity when illumination is also a learned variable. Example: a helmet visor was wrongly identified as metal during optimization.

2. **Cross-Environment Joint Optimization**:

    - **Function**: Breaks the albedo-lighting ambiguity through appearance constraints under multiple lighting environments.
    - **Mechanism**: The k sets of learnable environment lightmaps are optimized independently but query the same set of material properties. The PBR shading function ensures that the same material produces physically plausible but different appearances under different lighting conditions. Joint optimization guides the lighting representations to converge to values close to the real distribution, preventing them from becoming a "residual dustbin". In experiments, dual environments (k=2) already yield significant improvements, with marginal returns for adding more environments.
    - **Design Motivation**: Analogous to multi-task learning—different lighting environments are different "tasks", and the shared material represents the shared "features". The physical constraints of the shading function ensure the correctness of the decomposition, rather than relying on artificial regularizers.

3. **HDR Post-Processing Strategy**:

    - **Function**: Ensures that learned lighting values are compatible with standard HDR environment maps.
    - **Mechanism**: The environment lightmaps are constrained to non-negative values (linear HDR space), with post-processing using only clipping and gamma correction. Complex tone mappers (like Reinhard or ACES) hinder optimization due to the introduction of non-linearities. This allows new HDR maps to directly replace the learned lightmaps without extra image normalization or albedo scaling.
    - **Design Motivation**: Previous methods (such as GS-IR, R3DG) often omit gamma correction—while NVS is unaffected, relighting degrades severely. Correct linear HDR processing gives the learned values clear physical meanings.

### Loss & Training

The total loss includes: standard 3DGS image reconstruction loss + specular saturation penalty + energy conservation regularization + normal consistency loss $\mathcal{L}_{\text{dn}} = \lambda\|\mathbf{n} - \hat{\mathbf{n}}\|^2$ (shortest-axis normal vs. depth-derived normal). Key point: learnable normal residuals are not used—cross-lighting consistency naturally improves normal estimation, avoiding normals overfitting to specular highlight shapes under a single lighting condition.

## Key Experimental Results

### Main Results (Relighting PSNR, average of 6 unseen lighting environments)

| Method | Training Setting | Relight Avg↑ | NVS Bridge↑ |
|------|---------|-------------|-------------|
| 3DGS-DR✧ | Dual-env | 21.78 | 24.89 |
| GS-IR✧ | Dual-env | 22.45 | 20.78 |
| R3DG✧ | Dual-env | 22.21 | 20.98 |
| GShader✧ | Dual-env | 21.17 | 23.34 |
| TensoIR✧ (w/ GT scaling) | Dual-env | 24.49 | 24.50 |
| TensoIR✧ (w/o GT scaling) | Dual-env | 23.11 | 23.50 |
| **ReCap✧** | **Dual-env** | **25.82** | **26.95** |

### Ablation Study

| Env Map Range | Tonemap | Gamma | Relight | NVS |
|--------------|---------|-------|---------|-----|
| [0,1]→LDR | ✗ | ✗ | 23.55 | 29.97 |
| [0,1]→LDR | ✗ | ✓ | 24.07 | 30.09 |
| [0,∞)→HDR | clip | ✗ | 22.69 | 32.36 |
| **[0,∞)→HDR** | **clip** | **✓** | **25.82** | **32.23** |
| [0,∞)→HDR | reinhard | ✓ | 23.13 | 29.79 |

### Key Findings

- **Gamma correction is critical for relighting** (+1.7 PSNR), yet many existing methods ignore it (since it does not affect NVS).
- **HDR space + simple clipping** significantly outperforms LDR or complex tone mappers—linear HDR preserves physical consistency.
- **Cross-environment supervision naturally improves normal estimation**—under a single lighting condition, highlight shapes are incorrectly embedded in the normals, leading to baked-in highlights during relighting; dual environments eliminate this overfitting.
- The gain is largest when moving from single to dual environments (~2.5 PSNR), with diminishing returns when further increasing the number of environments.
- ReCap outperforms TensoIR (which requires GT scaling) without requiring any GT albedo scaling.

## Highlights & Insights

- **An inverse rendering perspective from multi-task learning**: Analogy-wise, modeling appearance under different lighting to multi-task learning is highly intuitive—the PBR shading function acts as a "task head" with physical constraints.
- **Eliminating the metallic parameter** seems counter-intuitive but actually removes a vital source of optimization ambiguity. Replacing it with a specular tint vector + saturation constraints maintains expressiveness while stabilizing optimization.
- **Normal improvement as a byproduct**: Rather than learning additional normal residuals, normals are naturally improved via multi-light consistency—avoiding normal overfitting to the specular highlight shapes of a single environment.

## Limitations & Future Work

- Requires extra captures of the same object under multiple lighting environments, increasing data acquisition costs.
- Does not support indirect illumination (inter-reflection), which remains limiting for highly reflective scenes.
- Feasibility of scaling to large outdoor scenes remains to be verified.
- Currently limited to a dual-environment setup; a mechanism to automatically determine the optimal number of environments is missing.

## Related Work & Insights

- **vs GShader**: Handcrafted shading function + single environment leads to limited relighting quality; ReCap significantly improves this via a split-sum variant + cross-environment supervision.
- **vs TensoIR**: Requires GT albedo scaling for relighting, which is impractical; ReCap requires absolutely no GT information.
- **vs R3DG**: Introduces ray tracing to handle indirect illumination but suffers from severe normal overfitting; ReCap's cross-environment constraints prevent such overfitting.
- **vs NeRD/NeRV**: NeRF-based relighting methods are computationally expensive; ReCap is based on 3DGS, enabling efficient and real-time performance.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-task perspective using cross-environment captures to resolve the albedo-lighting ambiguity is simple yet powerful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Expanded benchmark covering both diffuse and specular objects, and complete ablations with clear comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation derivation and complete technical details.
- Value: ⭐⭐⭐⭐ High practicality, providing a reliable material-lighting decoupling solution for 3DGS relighting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](../../CVPR2026/3d_vision/lumimotion_gaussian_relighting_dynamics.md)
- [\[CVPR 2025\] 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)
- [\[CVPR 2025\] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)
- [\[CVPR 2025\] Generative Multiview Relighting for 3D Reconstruction under Extreme Illumination Variation](generative_multiview_relighting_for_3d_reconstruction_under_extreme_illumination.md)

</div>

<!-- RELATED:END -->
