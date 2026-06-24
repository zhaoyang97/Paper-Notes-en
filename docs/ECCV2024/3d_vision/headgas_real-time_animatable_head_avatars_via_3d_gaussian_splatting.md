---
title: >-
  [Paper Note] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] Proposes HeadGaS, which equips each 3D Gaussian primitive with a learnable latent feature base, linearly blends features using expression parameters, and predicts expression-dependent color and opacity via an MLP. This design achieves **real-time (250+ fps)** and high-quality animatable head reconstruction, outperforming baselines in PSNR by approximately 2 dB.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Head Animation"
  - "Animatable Head Modeling"
  - "Expression Transfer"
  - "Real-Time Rendering"
date: 2026-05-08
content_hash: c13b22b1e1136699
---

# HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting

**Conference**: ECCV 2024  
**arXiv**: [2312.02902](https://arxiv.org/abs/2312.02902)  
**Code**: No public code  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Head Animation, Animatable Head Modeling, Expression Transfer, Real-Time Rendering

## TL;DR

Proposes HeadGaS, which equips each 3D Gaussian primitive with a learnable latent feature base, linearly blends features using expression parameters, and predicts expression-dependent color and opacity via an MLP. This design achieves **real-time (250+ fps)** and high-quality animatable head reconstruction, outperforming baselines in PSNR by approximately 2 dB.

## Background & Motivation

**Background**: Animatable 3D head reconstruction is a core technology for digital humans, AR/VR, and teleconferencing. NeRF-based methods (e.g., INSTA, NeRFBlendShape) suffer from a trade-off between quality and speed, reaching interactive frame rates of only 10-15 fps.

**Limitations of Prior Work**: NeRF methods are limited in rendering speed due to the dense sampling of volume rendering; explicit methods (meshes, point clouds) provide stronger geometric constraints but struggle to maintain photorealism (e.g., INSTA exhibits triangular mesh artifacts).

**Key Challenge**: The original design of 3DGS is a static scene representation, which does not support expression-driven dynamic appearance changes. Intuitively, one should move Gaussian positions to model dynamics, but this complicates optimization.

**Goal**: How to extend 3DGS to an animatable head representation while maintaining real-time rendering speed and high fidelity.

**Key Insight**: Inspired by traditional blendshape models, rather than moving Gaussian primitives, facial dynamics are represented by changing their **opacity and color**—effectively achieving motion through "over-representation."

**Core Idea**: Each Gaussian carries a learnable feature base, and expression parameters serve as blending weights for linear combination in the latent space, which are then passed through a lightweight MLP to output expression-dependent color and opacity.

## Method

### Overall Architecture

With a monocular video input, head poses and expression parameters (compatible with both FLAME and FaceWarehouse) are obtained via pre-processing. The core of the model is a **feature-augmented set of 3D Gaussians**:

$$\mathcal{G}_a = (\boldsymbol{\Sigma}, \boldsymbol{\mu}, \mathbf{F})$$

where $\mathbf{F} \in \mathbb{R}^{B \times f_{dim}}$ is the learnable feature base ($B$=expression dimension, $f_{dim}$=feature dimension). During rendering, expression parameters $\mathbf{e}_i$ are blended with the feature base to generate frame-specific features, which are fed into an MLP to predict color and opacity, and then rendered into images via standard tile-based rasterization.

### Key Designs

1. **Feature Blending**: Each Gaussian possesses a feature base $\mathbf{F} \in \mathbb{R}^{B \times f_{dim}}$. Given the expression weights $\mathbf{e}_i \in \mathbb{R}^B$ of the $i$-th frame, the frame-specific feature is obtained via linear blending:

$$\mathbf{f}_i = \mathbf{F}^T \mathbf{e}_i + \mathbf{f}_0$$

where $\mathbf{f}_0$ is a bias term. This is similar to traditional blendshapes but blended in the latent feature space rather than the geometric space. Here, $B=52$ (the first 52 expression coefficients of FLAME) and $f_{dim}=32$. Design Motivation: Allowing each Gaussian to independently learn dynamic features is more expressive than requiring a single MLP to learn the global dynamics of all Gaussians.

2. **Lightweight MLP for Color and Opacity Prediction**: The frame-specific feature $\mathbf{f}_i$ and the positional encoding $\psi(\boldsymbol{\mu})$ are fed into a two-layer MLP (with 64-channel hidden layers):

$$\mathbf{c}_i, \alpha_i = \phi(\mathbf{f}_i, \psi(\boldsymbol{\mu}))$$

The output consists of color $\mathbf{c}_i \in \mathbb{R}^{3(k+1)^2}$ (SH coefficients, $k=3$) and opacity $\alpha_i \in [0,1]$ (sigmoid constrained). The MLP is extremely small so as not to affect real-time rendering.

3. **Modeling Motion via Opacity Variation**: Key observation (Fig. 2) — there are **overlapping Gaussian primitives** in dynamic areas of the face, which alternately become visible/invisible under different expressions. For example, when closing the mouth, the opacity of lip Gaussians is high; when opening the mouth, these Gaussians become transparent, and another set of Gaussians in the jaw area emerges. This "over-representation" strategy avoids optimization difficulties introduced by explicit 3D displacements.

4. **Validation of Other Non-viable Alternatives**:

    - Direct blending of explicit parameters (without MLP): Severe artifacts appear in dynamic regions
    - Predicting position/rotation offsets $\Delta(\mu, R)$: Introducing 3D motion dimensions into the heuristic optimization of 3DGS leads to geometric inconsistencies
    - Expression vectors as MLP conditioning only (without blending): A single MLP struggles to learn the global dynamics of all Gaussians

### Loss & Training

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_1(I_r, I_{gt}) + \lambda_s \mathcal{L}_{\text{SSIM}}(I_r, I_{gt}) + \lambda_p \mathcal{L}_p(I_r, I_{gt})$$

- $\lambda_1=0.8$, $\lambda_s=0.2$, $\lambda_p=0.1$
- The perceptual loss $\mathcal{L}_p$ is based on the VGG network, activated after 10K iterations, and computed only within the head bbox region
- Initialization: 2500 Gaussian center points (subsampled from 3DMM mesh vertices), with the feature base $\mathbf{F}$ initialized to zero
- Densification is executed between 500 and 15K iterations, with 50K total training iterations, taking about 1 hour on a single V100 GPU

## Key Experimental Results

### Main Results

| Method | Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | Rendering Time (s)↓ |
|------|--------|-------|-------|--------|-------------|
| NHA | INSTA | 26.99 | 0.942 | 0.043 | 0.63 |
| INSTA | INSTA | 28.61 | 0.944 | 0.047 | 0.05 |
| NeRFBlendShape | INSTA | 30.52 | 0.955 | 0.056 | 0.10 |
| PointAvatar | INSTA | 30.68 | 0.952 | 0.058 | 0.1-1.5 |
| **HeadGaS** | **INSTA** | **32.50** | **0.971** | **0.033** | **0.004** |
| NeRFBlendShape | NBS | 34.34 | 0.970 | 0.031 | 0.10 |
| **HeadGaS** | **NBS** | **36.66** | **0.976** | **0.026** | **0.004** |

HeadGaS leads by about 2 dB in PSNR on the INSTA and NBS datasets, with a rendering speed-up of 10-25×.

### Ablation Study

| Variant | L2↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-----|-------|-------|--------|
| Ours w/o blending | 0.0012 | 30.28 | 0.955 | 0.041 |
| Ours w/ $\Delta(\mu,R)$ | 0.0014 | 29.83 | 0.953 | 0.045 |
| Ours change all | 0.0014 | 29.65 | 0.951 | 0.041 |
| Ours w/o MLP | 0.0009 | 32.08 | 0.968 | 0.033 |
| Ours w/o $\mathcal{L}_p$ | 0.0008 | 32.11 | 0.969 | 0.046 |
| **Ours (Full)** | **0.0008** | **32.50** | **0.971** | **0.033** |

### Key Findings

- **Modeling dynamics via opacity > Positional deformation**: Changing color/opacity (PSNR 32.50) is far superior to moving Gaussian positions (29.83), as spatial motion introduces geometric inconsistencies under the heuristic optimization mechanism of 3DGS.
- **Blending in feature space > Blending explicit parameters**: Latent feature blending + MLP decoding avoids the representational limits of directly controlling color/opacity using expression weights.
- **Role of perceptual loss**: Removing $\mathcal{L}_p$ degrades LPIPS from 0.033 to 0.046, with the quality loss mainly occurring in fine textures.

## Highlights & Insights

- **Counter-intuitive yet effective dynamic modeling**: Instead of moving Gaussians, letting Gaussians "appear/disappear" to represent motion is simple yet highly effective.
- **3DMM-decoupled design**: Not tied to a specific parametric model; validated compatibility with two different 3DMMs (FLAME and FaceWarehouse).
- **Extreme speed**: 250 fps at 512² resolution, which is 25-250× faster than interactive NeRF methods.
- **Cross-subject expression transfer**: Allows driving one person's model with another person's expression parameters without additional training.

## Limitations & Future Work

- **Memory consumption**: Storing a feature base of $B \times f_{dim}$ floats for each Gaussian leads to significant memory overhead when the expression dimension $B$ is large.
- **Dependence on head tracking quality**: Pose/expression estimation errors from the tracker propagate directly to the rendering results.
- **Training data coverage limitations**: If certain expressions are only observed from the front, rendering those expressions from side angles will fail.
- Compression of the feature base remains unexplored; it could be combined with HAC-like methods to further reduce model size.
- Can be scaled to more complex dynamic regions such as full-body or hands.

## Related Work & Insights

- **NeRFBlendShape**: Also uses expression parameter blending, but blends multi-level hash grid fields instead of individual Gaussian features.
- **PointAvatar**: Deformable point cloud representation, with stronger geometric constraints but lacking photorealism.
- **INSTA**: Based on FLAME mesh deformation + InstantNGP, reaches near-real-time speed but exhibits mesh artifacts.
- **4D-GS / Dynamic 3DGS**: General dynamic scene methods, whereas HeadGaS features a more streamlined and specialized design for heads.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The insight of "modeling facial dynamics via opacity variation instead of geometric deformation" is highly original and counter-intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 3 datasets, comparisons with 8 baselines, comprehensive ablations, and cross-subject transfer experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, and the visualization of opacity changes (Fig 2) is highly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — Enables real-time driving of digital avatars, directly applicable to AR/VR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [\[ECCV 2024\] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)
- [\[ICLR 2026\] FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](../../ICLR2026/3d_vision/fastgha_generalized_few-shot_3d_gaussian_head_avatars_with_real-time_animation.md)
- [\[ECCV 2024\] TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)
- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)

</div>

<!-- RELATED:END -->
