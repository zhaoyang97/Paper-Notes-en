---
title: >-
  [Paper Note] Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps
description: >-
  [ICLR 2026][3D Vision][shadow generation] This paper proposes Light-Geometry Interaction (LGI) maps, a 2.5D representation encoding light-occlusion relationships derived from monocular depth estimation. Embedded within a…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "shadow generation"
  - "relighting"
  - "light-geometry interaction"
  - "bridge matching"
  - "monocular depth"
date: 2026-05-08
content_hash: b35ac22f0526388e
---

# Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps

**Conference**: ICLR 2026
**arXiv**: [2602.21820](https://arxiv.org/abs/2602.21820)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: shadow generation, relighting, light-geometry interaction, bridge matching, monocular depth

## TL;DR

This paper proposes Light-Geometry Interaction (LGI) maps, a 2.5D representation encoding light-occlusion relationships derived from monocular depth estimation. Embedded within a bridge matching generative framework, LGI maps enable joint modeling of shadow generation and object relighting, achieving state-of-the-art performance on both synthetic and real images.

## Background & Motivation

Shadow generation and relighting are critical for applications such as virtual product placement, augmented reality, and image editing. Traditional approaches rely on full 3D reconstruction and ray tracing, which are computationally expensive and infeasible under single-view settings. Recent generative methods based on diffusion models and bridge matching can synthesize shadows from RGB inputs, but the absence of physical constraints frequently leads to the following artifacts:

- **Floating shadows**: shadows geometrically inconsistent with the object
- **Lighting inconsistency**: relighting direction contradicts shadow direction
- **Unreasonable shadow geometry**: failure under complex occlusion scenarios

More critically, existing methods treat shadow generation and relighting as **independent tasks**, ignoring their intrinsic coupling—accurate modeling must simultaneously account for direct illumination, secondary reflections, and interreflections.

## Core Problem

How can the interaction between lighting and geometry be efficiently encoded from monocular depth alone in a single-view setting, and how can this encoding serve as a physical prior embedded in a generative model for joint shadow generation and relighting?

## Method

### Overall Architecture

Built upon the Latent Bridge Matching (LBM) framework, the method transforms a shadow-free image $x_0$ into a shadowed image $x_1$. The central contribution is the introduction of LGI maps as conditioning signals that provide illumination-aware occlusion cues. The encoder and decoder are taken from pretrained Stable Diffusion XL and remain frozen during training.

The drift network $v_\theta$ is conditioned on $c = \{c^l, c^m\}$, where $c^l$ denotes global lighting parameters (light color, radius, distance, intensity, azimuth, and elevation angle), and $c^m$ denotes the LGI maps.

### LGI Map Generation (Five-Step Pipeline)

**Step 1 — Depth Estimation**: An off-the-shelf monocular depth estimator is used to obtain depth map $D$, which is then rescaled to be consistent with the light source coordinate system.

**Step 2 — 2D-to-3D Lifting**: Each pixel is lifted into 3D space via inverse camera projection:

$$p = D(u,v) \cdot K^{-1} [u, v, 1]^\top$$

**Step 3 — Ray Sampling**: From each 3D point $p$, a ray is cast toward the light source $l$. $N=16$ points are uniformly sampled along the ray and reprojected onto the image plane to retrieve corresponding depth values.

**Step 4 — Elevation Difference Computation**: For each sampled point, the difference $e^d_n = e^s_n - e^l$ is computed between the surface elevation angle $e^s_n$ and the ray elevation angle $e^l$. If the surface elevation in a given direction exceeds the ray elevation, the point is considered occluded and lies within a shadow region.

**Step 5 — Three-Channel LGI Map Construction**:

- $c^m_1 = \min e^d_n$: minimum elevation difference, indicating the onset of occlusion
- $c^m_2 = \max e^d_n$: maximum elevation difference, indicating the end of occlusion
- $c^m_3 = e^d_{i^*}$, where $i^* = \arg\min |e^d_n|$: the difference with the smallest absolute value, indicating the most likely direct occlusion point

LGI values are naturally bounded within $(-\pi, \pi)$, which promotes stability of the network inputs.

### Loss & Training

A weighted L1 loss replaces the standard pixel-wise loss. Shadow-changing regions are emphasized using a luminance-change threshold $\tau=0.01$ combined with a dilation operation:

$$\mathcal{L}_x(\hat{x}_1, x_1) = \frac{1}{M}\sum_{m=1}^M w^{(m)} \cdot |x_1^{(m)} - \hat{x}_1^{(m)}|$$

The final loss combines latent space matching with the weighted pixel loss, with weight $\lambda=10$.

### Extension to Image Harmonization

The framework is also extended to image harmonization: an additional lighting estimation network is introduced to infer lighting conditions from composite images. Since LGI maps are fully differentiable, shadow masks enable self-supervised lighting estimation.

### ShadRel Dataset

The paper introduces the first large-scale dataset targeting the joint shadow-relighting task:

- **817K** virtual objects created by professional 3D artists
- Rendered using Blender Cycles path tracing
- Diverse materials including glossy, metallic, and transparent surfaces (based on principled BSDF)
- Each object is rendered under 4 random camera viewpoints × 5 lighting configurations = 20 target images
- Covers challenging scenarios including soft shadows, reflections, transparency, and interreflections

## Key Experimental Results

### Joint Shadow Generation and Relighting (ShadRel Dataset)

| Method | Overall RMSE↓ | Overall SSIM↑ | Shadow BER↓ | Shadow IoU↑ | Object RMSE↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| LBM | 0.0417 | 0.7148 | 0.0847 | 0.7166 | 0.0298 |
| **Ours** | **0.0334** | **0.7227** | **0.0588** | **0.8096** | **0.0282** |

Shadow region RMSE is reduced from 0.1543 to 0.0898 (**42% improvement**); BER decreases from 0.1549 to 0.1103.

### Clean-Background Shadow Generation (CSG Benchmark)

The proposed method outperforms CSG on IoU across all three control tracks (0.821 vs. 0.818, 0.798 vs. 0.780, 0.785 vs. 0.776).

### Image Harmonization (DESOBAv2)

Overall performance is comparable to the best-performing method SGDGP, while achieving higher accuracy in shadow regions (Local RMSE: 44.753 vs. 46.713).

### Ablation Study

- LGI maps are the most critical component; removing them degrades Shadow BER from 0.0588 to 0.0940.
- Substituting raw depth maps for LGI maps yields only marginal improvement (−LGI+Depth: BER 0.0932 vs. baseline 0.1012).
- The full three-channel LGI outperforms using only the third channel (BER 0.0588 vs. 0.0670).
- Switching to DepthAnythingV2 or ground-truth depth causes negligible change in results, demonstrating robustness to the choice of depth estimator.
- Computational overhead is nearly negligible: parameter count increases by only 0.0004% and FLOPs by 0.0011%.

## Highlights & Insights

1. **Elegant design of LGI maps**: The core idea of ray tracing is distilled into a differentiable 2.5D representation. Light-occlusion relationships are encoded without full 3D reconstruction, combining physical intuition with computational efficiency.
2. **Joint modeling paradigm**: For the first time, shadow generation and relighting are unified within a single framework, capturing the coupled effects of direct illumination, secondary reflections, and interreflections.
3. **Strong generalization**: Trained exclusively on synthetic data, the model performs well on real images (including portraits) without any real-world fine-tuning.
4. **Computational efficiency**: The LGI module introduces virtually zero additional computational cost and naturally scales to multi-object and multi-light-source scenarios.

## Limitations & Future Work

- Inherent limitations of 2.5D depth representation: depth information in occluded regions is unavailable, leading to ambiguous shadows (as illustrated in Fig. 3d of the paper).
- Training data is entirely synthetic; while generalization is acceptable, performance may degrade in extreme real-world scenarios.
- Monocular depth estimation lacks metric scale and relies on consistency assumptions with light source coordinates.
- The current formulation supports only point light source modeling and has not been extended to area lights or environment illumination.
- The image harmonization extension requires an additional lighting estimation network, increasing overall system complexity.

## Related Work & Insights

| Dimension | CSG / LBM | SGDGP | SwitchLight | Ours |
|------|-----------|-------|-------------|------|
| Shadow Generation | ✓ | ✓ | ✗ | ✓ |
| Relighting | ✗ | ✗ | ✓ | ✓ |
| Joint Modeling | ✗ | ✗ | ✗ | ✓ |
| Geometric Prior | None / 2D template | Bounding box + template | None | LGI maps (2.5D) |
| Physical Constraints | Weak | Moderate | Weak | Strong |
| Real Image Generalization | Moderate | Good | Primarily portraits | Good (including portraits) |

The core idea behind LGI maps—reducing the ray tracing process to statistics over elevation angle differences—is transferable to other tasks requiring illumination modeling, such as intrinsic decomposition and lighting estimation. The three-channel design (min/max/closest) elegantly encodes the degree of occlusion uncertainty, providing an effective strategy for handling 2.5D depth ambiguity. The fully differentiable design allows natural integration into any end-to-end framework, not limited to bridge matching. The ShadRel dataset addresses the absence of training data for joint shadow-relighting tasks and serves as an important benchmark for future research.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — LGI maps offer a novel representation; the joint modeling paradigm constitutes a clear contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-benchmark comparisons, comprehensive ablations, and qualitative analysis on real images.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear presentation, complete mathematical derivations, and intuitive figures.
- **Value**: ⭐⭐⭐⭐ — Strong practical utility, computationally efficient, and a valuable dataset contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LiTo: Surface Light Field Tokenization](lito_surface_light_field_tokenization.md)
- [\[ICLR 2026\] Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation](ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](../../CVPR2026/3d_vision/lumimotion_gaussian_relighting_dynamics.md)
- [\[AAAI 2026\] Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues](../../AAAI2026/3d_vision/geometry_meets_light_leveraging_geometric_priors_for_universal_photometric_stere.md)
- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)

</div>

<!-- RELATED:END -->
