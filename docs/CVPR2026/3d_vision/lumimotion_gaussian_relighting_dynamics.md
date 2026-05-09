---
title: >-
  [Paper Note] LumiMotion: Improving Gaussian Relighting with Scene Dynamics
description: >-
  [CVPR 2026][3D Vision][Inverse Rendering] LumiMotion is the first Gaussian-based inverse rendering method that leverages scene dynamics (motion regions) as supervision signals to improve material-lighting decomposition. Through static-dynamic separation and motion-revealed appearance changes, it achieves a 23% improvement in albedo LPIPS and a 15% improvement in relighting LPIPS.
tags:
  - CVPR 2026
  - 3D Vision
  - Inverse Rendering
  - 2D Gaussian Splatting
  - Dynamic Scenes
  - Material Estimation
  - Relighting
date: 2026-05-08
content_hash: 9993eb7ac693fc1e
---

# LumiMotion: Improving Gaussian Relighting with Scene Dynamics

**Conference**: CVPR 2026
**arXiv**: [2604.10994](https://arxiv.org/abs/2604.10994)
**Code**: [https://joaxkal.github.io/LumiMotion/](https://joaxkal.github.io/LumiMotion/)
**Area**: 3D Vision
**Keywords**: Inverse Rendering, 2D Gaussian Splatting, Dynamic Scenes, Material Estimation, Relighting

## TL;DR
LumiMotion is the first Gaussian-based inverse rendering method that leverages scene dynamics (motion regions) as supervision signals to improve material-lighting decomposition. Through static-dynamic separation and motion-revealed appearance changes, it achieves a 23% improvement in albedo LPIPS and a 15% improvement in relighting LPIPS.

## Background & Motivation

**Background**: Inverse rendering aims to recover geometry, material, and lighting from images. Existing Gaussian Splatting methods (R3DG, IRGS, GI-GS) primarily target static scenes and tend to confuse shadows with material color under strong directional lighting.

**Limitations of Prior Work**: In static scenes, it is difficult to distinguish whether a dark region is caused by a shadow or by the material's intrinsic color, due to the lack of observations of the same surface under varying lighting conditions. Existing dynamic scene methods are either restricted to human avatars or require known or multi-illumination training setups.

**Key Challenge**: Accurate material-lighting separation requires multi-illumination observations of the same surface, yet real-world captures typically provide only a single lighting condition.

**Goal**: Exploit object motion within a scene (e.g., moving shadows, illumination changes on moving objects) as a natural multi-illumination supervision signal.

**Core Idea**: Motion reveals the appearance of the same surface under different lighting conditions, providing stronger constraints for material-lighting decomposition.

## Method

### Overall Architecture
A two-stage approach: Stage 1 trains a dynamic 2DGS representation (learning geometry, static-dynamic separation, and time-varying color) → Stage 2 freezes the geometry and deformation network, then jointly optimizes material parameters (albedo, roughness) and environment lighting, using ray tracing to compute visibility and indirect illumination.

### Key Designs

1. **Binary Concrete Static-Dynamic Separation**:

    - Function: Explicitly separates static and dynamic Gaussians during Stage 1.
    - Mechanism: An auxiliary variable $P$ is introduced per Gaussian. A sample $\tilde{P}$ is drawn from the Binary Concrete distribution (a continuous relaxation of the Bernoulli distribution) and multiplied with the deformation network output. Gaussians with $\tilde{P} \approx 0$ remain static, while those with $\tilde{P} \approx 1$ follow the deformation field.
    - Design Motivation: Moving shadows can be interpreted either as color changes or as Gaussian movement/disappearance — the latter prevents stable albedo assignment in Stage 2. It is therefore essential that shadow regions be explained by static Gaussians with time-varying color.

2. **Multiplicative Time-Varying Color Model**:

    - Function: Models moving shadows and illumination changes on dynamic objects.
    - Mechanism: The color is defined as $c' = c \cdot (1 - \Delta c)$, where the multiplicative form simulates the effect of lighting on surfaces (consistent with the rendering equation). The canonical color $c$ approximates a pseudo-albedo and serves as the initial estimate for Stage 2.
    - Design Motivation: Additive color changes are physically inconsistent with lighting models; multiplicative changes are more natural and easier to regularize.

3. **Hierarchical Sampling Ray Tracing**:

    - Function: Efficiently computes visibility and indirect illumination for inverse rendering.
    - Mechanism: In Stage 2, the geometry is frozen and albedo, roughness, and normals are rasterized into a G-buffer. Ray tracing with hierarchically sampled environment light directions is used to compute visibility $V$ and indirect illumination $L_{\text{ind}}$, under the Disney BRDF model.
    - Design Motivation: Shadows in dynamic scenes vary over time; ray tracing accurately captures these changes and provides correct lighting information for material estimation.

### Loss & Training
Stage 1: Reconstruction loss + normal consistency + depth distortion + foreground mask BCE + static-dynamic separation regularization (encouraging $P \to 0$) + color change regularization. Stage 2: L1 loss under the rendering equation + albedo smoothness regularization.

## Key Experimental Results

### Main Results

| Scene / Metric | LumiMotion | Prev. SOTA (IRGS) | Gain |
|---|---|---|---|
| Albedo LPIPS | Best | 2nd | −23% |
| Relighting LPIPS | Best | 2nd | −15% |
| Relighting PSNR | Best | 2nd | Significant |

### Ablation Study

| Configuration | Relighting PSNR | Note |
|---|---|---|
| Full (dynamic) | Best | Leverages dynamic information |
| Static baseline | Worse | Shadows baked into albedo |
| w/o static-dynamic separation | Degraded | Dynamic Gaussians interfere with albedo |

### Key Findings
- In dynamic scenes, LumiMotion successfully removes shadows from albedo, whereas static methods bake shadows into the albedo.
- On static and dynamic versions of the same scene, the dynamic version consistently yields better inverse rendering results.
- Binary Concrete separation is critical for accurate albedo estimation.

## Highlights & Insights
- **Motion as Supervision**: The core observation is highly insightful — motion naturally provides samples of the same surface under different lighting conditions, serving as a source of "free" multi-illumination data.
- **Controlled Benchmark Release**: A new synthetic benchmark with paired static/dynamic versions enables systematic evaluation of the impact of dynamics on inverse rendering for the first time.

## Limitations & Future Work
- Assumes static lighting; not applicable to scenes where illumination also changes over time.
- Requires sufficient motion regions in the scene to provide effective supervision.
- Indirect illumination modeling remains relatively simplified.

## Related Work & Insights
- **vs. IRGS**: IRGS handles only static scenes and has limited shadow removal capability.
- **vs. Relightable Neural Actor**: Restricted to human avatars and requires known lighting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to leverage scene dynamics for improved inverse rendering; highly insightful observation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on both synthetic and real data with controlled static/dynamic comparisons.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ Opens a new direction for dynamic inverse rendering.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](noderf_neural_ode_nerf_continuous_spacetime_dynam.md)
- [\[ICLR 2026\] Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics](../../ICLR2026/3d_vision/improving_long-range_interactions_in_graph_neural_simulators_via_hamiltonian_dyn.md)
- [\[CVPR 2026\] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)
- [\[CVPR 2026\] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator](gaussfusion_improving_3d_reconstruction_in_the_wild_with_a_geometry-informed_vid.md)
- [\[NeurIPS 2025\] MetaGS: A Meta-Learned Gaussian-Phong Model for Out-of-Distribution 3D Scene Relighting](../../NeurIPS2025/3d_vision/metags_a_meta-learned_gaussian-phong_model_for_out-of-distribution_3d_scene_reli.md)

<!-- RELATED:END -->
