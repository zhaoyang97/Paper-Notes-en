---
title: >-
  [Paper Note] SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] SpectroMotion based on the 3DGS framework models dynamic objects via a deformable Gaussian MLP and time-varying illumination effects via a deformable reflection MLP. Combined with a canonical environment map and a coarse-to-fine three-stage training strategy, it achieves high-quality 3D reconstruction and real-time rendering of dynamic specular scenes for the first time.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Dynamic Scenes"
  - "Specular Reflection"
  - "Environment Illumination"
  - "Deformable Gaussians"
date: 2026-05-08
content_hash: f01a24e704cfa292
---

# SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes

**Conference**: CVPR 2025  
**arXiv**: [2410.17249](https://arxiv.org/abs/2410.17249)  
**Code**: None  
**Area**: 3D Vision / Dynamic Scene Reconstruction  
**Keywords**: 3D Gaussian Splatting, Dynamic Scenes, Specular Reflection, Environment Illumination, Deformable Gaussians

## TL;DR

SpectroMotion based on the 3DGS framework models dynamic objects via a deformable Gaussian MLP and time-varying illumination effects via a deformable reflection MLP. Combined with a canonical environment map and a coarse-to-fine three-stage training strategy, it achieves high-quality 3D reconstruction and real-time rendering of dynamic specular scenes for the first time.

## Background & Motivation

1. **Background**: 3D Gaussian Splatting (3DGS) has made breakthrough progress in novel view synthesis of static scenes, while methods such as Deformable 3DGS have extended it to dynamic scenes. On the other hand, works like GaussianShader and GS-IR address specular reflections in static scenes. However, simultaneously handling the intersection of dynamic motion and specular reflections remains an unresolved challenge.
2. **Limitations of Prior Work**: (1) Dynamic methods such as Deformable 3DGS and 4DGS use spherical harmonics (SH) to model color, which cannot accurately represent view-dependent specular reflections; (2) Specular reflection methods like GaussianShader and GS-IR only handle static scenes and cannot cope with object motion and time-varying illumination; (3) Although NeRF-DS specifically targets dynamic specular scenes, it is based on NeRF volume rendering, which is slow and yields limited quality.
3. **Key Challenge**: The specular appearance of objects in dynamic scenes changes not only with the viewpoint but also with object motion and temporal variations in environment illumination, presenting a triple coupling of geometric deformation, material properties, and lighting conditions.
4. **Goal**: To unify the modeling of dynamic object motion and specular reflection effects within the 3DGS framework, achieving high-quality rendering and reliable geometry/material decomposition.
5. **Key Insight**: The final color is decomposed into diffuse and specular components, which are modeled by different mechanisms; a phased training strategy is used to progressively introduce geometric deformation, normal optimization, and specular reflection capabilities.
6. **Core Idea**: Combining a deformable Gaussian MLP (handling object motion) + a canonical environment map (time-invariant lighting baseline) + a deformable reflection MLP (time-varying lighting deviation), and adopting a coarse-to-fine three-stage training strategy (static $\rightarrow$ dynamic $\rightarrow$ specular) to stably optimize all components.

## Method

### Overall Architecture

The input is a monocular video sequence, and the output is a 3DGS representation of the dynamic scene. 3D Gaussians are defined in the canonical space, with their position, rotation, and scale offsets at each timestep predicted by a deformable Gaussian MLP. The color representation is decomposed into $c_{\text{final}} = c_{\text{diffuse}} + c_{\text{specular}}$, where the diffuse component is represented by zero-order spherical harmonics (SH), and the specular component obtains the base reflection color by querying the environment map, followed by predicting time-varying lighting offsets via a deformable reflection MLP. Training is performed progressively in three stages.

### Key Designs

1. **Deformable Gaussian MLP**:
    - **Function**: Models the dynamic motion of objects in the scene, predicting the deformation (position, rotation, and scale offsets) of each 3D Gaussian at different timesteps.
    - **Mechanism**: Following the design of Deformable 3DGS, the spatial coordinates and temporal information of the 3D Gaussians are taken as input. After passing through 8 fully connected (FC) layers (256-dimensional hidden layers with ReLU activation) to yield a 256-dimensional feature vector, three separate branches output the offsets for position, rotation, and scale, respectively. The 4th layer adopts a skip connection (similar to NeRF) to concatenate the input with intermediate features.
    - **Design Motivation**: Decoupling dynamic modeling from the color representation allows subsequent specular reflection modeling to proceed on a stable geometric foundation. It automatically distinguishes between dynamic and static objects without requiring mask supervision.

2. **Canonical Environment Map + Deformable Reflection MLP**:
    - **Function**: Models the time-invariant base lighting and time-varying illumination effects, respectively.
    - **Mechanism**: The environment map uses learnable cubemap parameters of size $6 \times 128 \times 128$ to represent the canonical (average/baseline) lighting conditions of the scene. Given the normal direction of the Gaussian and the camera viewpoint, the reflection direction is computed via physical reflection equations, which is then used to query the environment map for the base specular color. The deformable reflection MLP learns the mapping from time to lighting offsets, capturing changes in reflective appearance caused by object motion. The final specular color integrates specular tint and roughness attributes.
    - **Design Motivation**: Decomposing the lighting into a time-invariant baseline and a time-varying deviation reduces learning difficulty. Since SH alone cannot accurately model specular highlights, the environment map coupled with a physical reflection model offers a more precise representation of view-dependent reflection effects.

3. **Coarse-to-Fine Three-Stage Training Strategy**:
    - **Function**: Solves the optimization coupling issue among dynamics, geometry, and specular reflection.
    - **Mechanism**: **Static Phase** (3k iterations): Trains standard 3DGS to stabilize static geometry. **Dynamic Phase** (6k iterations): Introduces the deformable Gaussian MLP, optimizing basic deformation during the first 3k iterations, and incorporating the normal loss $\mathcal{L}_{\text{normal}}$ to optimize normals and depth simultaneously during the next 3k iterations. **Specular Phase** (31k iterations): Switches the SH color to $c_{\text{final}}$, freezes the deformable Gaussian MLP and most parameters to optimize only zero-order SH, specular tint, and roughness, and unfreezes all parameters after 6k iterations. The first 2k iterations of this phase optimize only the canonical environment map, after which the deformable reflection MLP is introduced. Total 40k iterations.
    - **Design Motivation**: Simultaneously optimizing all components would cause incomplete color representations to disrupt the learned geometry. The phased strategy ensures that each newly introduced component stands on a stable foundation. Learning dynamics before reflection avoids optimization conflicts between dynamic motion and specular effects.

### Loss & Training

- Standard 3DGS reconstruction loss ($L_1$ + SSIM)
- Normal consistency loss $\mathcal{L}_{\text{normal}}$: constrains the rendered normal to align with the depth-derived normal
- Adam optimizer, 40,000 total iterations
- Adaptive Gaussian densification and pruning strategies

## Key Experimental Results

### Main Results

| Method | NeRF-DS Mean PSNR↑ | Mean SSIM↑ | Mean LPIPS↓ |
|------|-------------------|-----------|------------|
| Deformable 3DGS | 19.66 | 0.5826 | 0.3181 |
| 4DGS | 18.09 | 0.4649 | 0.4078 |
| GaussianShader | 14.98 | 0.3681 | 0.6121 |
| GS-IR | 15.05 | 0.3678 | 0.5856 |
| NeRF-DS | 18.74 | 0.5151 | 0.4337 |
| HyperNeRF | 16.23 | 0.5007 | 0.4420 |
| **SpectroMotion** | **20.08** | **0.5909** | **0.3094** |

### Ablation Study

| Scene | SpectroMotion PSNR | Deformable 3DGS PSNR | Gain | Description |
|------|-------------------|---------------------|------|------|
| As | 24.51 | 24.14 | +0.37 | Scenes with weaker specular effects |
| Bell | 19.60 | 19.42 | +0.18 | Contains strong specular reflection |
| Cup | 20.13 | 20.10 | +0.03 | Small gap but still optimal |
| Plate | 16.53 | 16.12 | +0.41 | Significant improvement in highly specular scenes |
| Press | 21.70 | 19.64 | +2.06 | Maximum improvement, complex specular dynamics |
| Sieve | 20.36 | 20.74 | -0.38 | The only scene outperformed by Deformable 3DGS |

### Key Findings

- In the specialized evaluation of dynamic specular objects (using dynamic specular masks generated via Track Anything), SpectroMotion comprehensively outperforms all methods.
- It automatically distinguishes between dynamic and static objects without mask supervision (verified by visualizing the deformation magnitude of the deformable Gaussian MLP).
- The diffuse/specular decomposition results are visually plausible—specular components map to smooth metallic surfaces.
- Most scenes contain <200k Gaussians, allowing real-time rendering at $\ge 30$ FPS.
- Training time is approximately 1-2 hours (RTX 4090), which is significantly faster than NeRF-DS.
- Limitation: Floaters can appear during drastic scene changes (e.g., hands entering/leaving the frame).

## Highlights & Insights

- **Unifying dynamics and specular reflection under the 3DGS framework for the first time**: Previously, dynamic 3DGS and specular 3DGS were studied as separate domains. SpectroMotion elegantly unifies both under a single framework, representing a major contribution in its own right.
- **Robustness of the phased training strategy**: Progressive training starting from static $\rightarrow$ dynamic $\rightarrow$ specular stages circumvent conflicts among multiple learning objectives. The design of the specular stage (freezing geometry first and then unfreezing) balances the optimization of newly introduced and existing components.
- **Decomposition strategy for time-varying illumination**: Factoring illumination into a canonical environment map (baseline) + a deformable reflection MLP (time-varying deviation) is a design that can be transferred to other tasks handling time-varying appearance.

## Limitations & Future Work

- Unable to handle drastic scene variations (such as new objects entering/leaving the scene), relying on stable foreground objects.
- Employs only monocular video, which may introduce ambiguities in geometrically complex regions.
- The environment map assumes global illumination, failing to model shadow variations caused by local occlusions.
- Future Improvements: Incorporating space-time voxel representations of 4DGS to handle severe motion, introducing physically-constrained material models (BRDF), and extending to multi-view inputs.

## Related Work & Insights

- **vs Deformable 3DGS**: Models only dynamics but uses SH to represent color, struggling with specular reflection. SpectroMotion incorporates a complete specular reflection modeling scheme on top of it, while retaining dynamic modeling capabilities.
- **vs NeRF-DS**: The only baseline previously addressing dynamic specular scenes, but based on NeRF volume rendering, which is slow (not real-time). SpectroMotion achieves real-time rendering using 3DGS with superior quality (PSNR +1.34, LPIPS -0.1243).
- **vs GaussianShader/GS-IR**: Models only static specular scenes. SpectroMotion demonstrates that their static specular representation schemes fail completely in dynamic scenes (PSNR only 14-15).

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to unify dynamics and specular reflection in 3DGS, though individual components (deformable MLP, environment map) are combinations of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐ Evaluated only on the NeRF-DS dataset, lacking comprehensive evaluation on other datasets like HyperNeRF.
- **Writing Quality**: ⭐⭐⭐⭐ The training strategy is described clearly, but there is limited formulation of the method.
- **Value**: ⭐⭐⭐⭐ Fills a gap in dynamic specular 3DGS reconstruction with clear application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Uncertainty-Aware 3D Reconstruction for Dynamic Underwater Scenes](../../ICLR2026/3d_vision/uncertainty-aware_3d_reconstruction_for_dynamic_underwater_scenes.md)
- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](../../NeurIPS2025/3d_vision/d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)

</div>

<!-- RELATED:END -->
