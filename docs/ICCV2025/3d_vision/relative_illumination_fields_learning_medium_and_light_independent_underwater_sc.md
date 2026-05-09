---
title: >-
  [Paper Note] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes
description: >-
  [3D Vision] This paper proposes Relative Illumination Fields (RIF), which models non-uniform illumination distributions in camera-local coordinates via an MLP and jointly optimizes a volumetric medium representation, enabling clean reconstruction of underwater scenes free from light source and medium effects.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: 8ddfdd842b9543ee
---

# Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2504.10024](https://arxiv.org/abs/2504.10024)
- **Code**: Not released
- **Area**: 3D Vision
- **Keywords**: Underwater NeRF, illumination field, scattering medium, joint optimization, color restoration

## TL;DR

This paper proposes Relative Illumination Fields (RIF), which models non-uniform illumination distributions in camera-local coordinates via an MLP and jointly optimizes a volumetric medium representation, enabling clean reconstruction of underwater scenes free from light source and medium effects.

## Background & Motivation

Underwater environments pose unique challenges for 3D reconstruction:

**Scattering and Attenuation**: Water absorbs and scatters light in a wavelength-dependent manner, effectively introducing an "extra object" between the scene and the camera.

**Dynamic Non-uniform Illumination**: At depths beyond a few tens of meters, sunlight is absent. Artificial light sources carried by underwater robots move co-jointly with the camera, producing intense non-uniform backscatter light cones.

**Limitations of Prior Work**:
   - Methods such as SeaThru-NeRF assume uniform illumination (analogous to a fog model) and cannot handle artificial light source scenarios.
   - DarkGS and similar methods require known and calibrated light source models, limiting practical applicability.
   - The closest prior work assumes a single point light source at the camera center, which is overly simplistic.

**Core Observation**: In co-moving light source scenarios, the illumination distribution remains constant in the camera-local coordinate frame. What matters is not the specific parameters of each light source, but the cumulative illumination received at a given point within the camera frustum.

## Method

### Overall Architecture

Built upon Nerfstudio/Nerfacto, the framework comprises three components: a global NeRF MLP, a local illumination field MLP, and a medium representation.

### 1. Local Illumination Field Representation

An MLP $\mathcal{F}^l_\Theta$ defined in the camera-local coordinate frame:

$$\boldsymbol{\alpha} = \mathcal{F}^l_\Theta(\phi_{\text{Hash}}(\boldsymbol{x}^c), \phi_{\text{SH}}(\boldsymbol{n}^c))$$

- $\boldsymbol{x}^c$: position of the sampled point in camera coordinates
- $\boldsymbol{n}^c$: surface normal direction in camera coordinates (derived from the density field gradient)
- $\boldsymbol{\alpha}$: illumination intensity factor (independently estimated per color channel)

Coordinate transforms: $\mathbf{x}^c = {}^c\mathbf{T}_w \cdot \mathbf{x}^w$, $\boldsymbol{n}^c = {}^c\mathbf{R}_w \cdot \boldsymbol{n}^w$

**Simplifying Assumptions**: BRDF is neglected (most natural underwater materials are assumed Lambertian); light source visibility is neglected (lights are typically close to the camera, and shadows fall on the back faces of objects).

### 2. Volumetric Medium Representation

The rendering equation is decomposed into object color and medium color:

$$\boldsymbol{C}(\boldsymbol{r}(t)) = \sum_i^N \boldsymbol{C}_i^{\text{obj}}(\boldsymbol{r}(t)) + \boldsymbol{C}_i^{\text{med}}(\boldsymbol{r}(t))$$

where:
- Object component: $\boldsymbol{C}_i^{\text{obj}} = T_i^{\text{obj}} \cdot T_i^{\text{attn}} \cdot (1 - e^{-\sigma_i^{\text{obj}}\delta_i}) \cdot \boldsymbol{c}_i^{\text{obj}}$
- Medium component: $\boldsymbol{C}_i^{\text{med}} = T_i^{\text{obj}} \cdot T_i^{\text{bs}} \cdot (1 - e^{-\boldsymbol{\sigma}^{\text{bs}}\delta_i}) \cdot \boldsymbol{c}^{\text{med}}$

Medium parameters $\boldsymbol{\sigma}^{\text{attn}}$ (attenuation coefficient), $\boldsymbol{\sigma}^{\text{bs}}$ (backscattering coefficient), and $\boldsymbol{c}^{\text{med}}$ (medium color) are treated as globally optimizable parameters.

### 3. Full Model

The illumination field and medium model are combined as:

$$\boldsymbol{C}(\boldsymbol{r}(t)) = \sum_i^N \boldsymbol{\alpha}_i (C_i^{\text{obj}}(\boldsymbol{r}(t)) + \boldsymbol{C}_i^{\text{med}}(\boldsymbol{r}(t)))$$

The illumination factor $\boldsymbol{\alpha}_i$ is applied jointly to both the object and medium color components.

### Loss & Training

An HDR-aware loss following the RawNeRF strategy is adopted:

$$\mathcal{L} = \sum_{\boldsymbol{r} \in \mathcal{R}} \|\frac{\hat{C}(\boldsymbol{r}(t)) - C(\boldsymbol{r}(t))}{\text{sg}(\hat{C}(\boldsymbol{r}(t))) + \epsilon}\|^2$$

where $\text{sg}(\cdot)$ denotes stop-gradient and $\epsilon = 10^{-3}$.

## Key Experimental Results

### Main Results: Co-moving Light and Medium Removal

Evaluated on five datasets (synthetic in-air, synthetic underwater, and real underwater) containing 1–4 co-moving light sources:
- **DarkGS dataset (in-air)**: Achieves clean scene recovery without any light source calibration, whereas DarkGS requires explicit calibration.
- **Beyond NeRF underwater dataset**: Successfully disentangles the light cone, water scattering, and object color.
- **Self-collected real underwater data**: Captured in a 1 m × 2 m water tank with a GoPro; illumination and medium effects are successfully removed.

### Ablation Study: Single-channel vs. Three-channel Illumination Field

| Configuration | Synthetic In-air L2↓ | Synthetic Underwater L2↓ | Real Underwater L2↓ |
|------|------|------|------|
| Single-channel $\alpha$ | 9.554 | 12.780 | 28.944 |
| **Three-channel $\boldsymbol{\alpha}$** | **10.143** | **11.879** | **30.077** |

### Key Findings
1. **Three-channel illumination field is superior underwater**: Because light undergoes wavelength-dependent attenuation before reaching the object, per-channel modeling is necessary.
2. **Single- and three-channel perform comparably in air**: Without medium attenuation, assuming all light sources share the same color suffices.
3. **No light source calibration required**: The illumination distribution is learned entirely from data, which is a key advantage.

## Highlights & Insights

1. **Elegant observation of "relative illumination"**: In co-moving light source scenarios, the illumination distribution in the camera-local coordinate frame remains invariant, greatly simplifying the problem.
2. **Fully calibration-free**: The method requires no prior knowledge of light source count, position, or intensity distribution, learning solely from multi-view observations.
3. **Modular design**: Removing the medium term makes the framework directly applicable to in-air low-light scenes.

## Limitations & Future Work

- Shadow effects (light source visibility) are neglected; performance degrades when shadows are significant.
- Regions that are never illuminated throughout the dataset cannot be recovered.
- A scale factor must be manually set per dataset to accommodate varying dynamic ranges.

## Related Work & Insights

- **Underwater NeRF**: SeaThru-NeRF, UW-NeRF
- **Low-light reconstruction**: DarkGS, RawNeRF
- **Relighting NeRF**: NeRFactor, S3-NeRF

## Rating

- Novelty: ⭐⭐⭐⭐ (The core observation underlying the local illumination field is highly elegant)
- Technical Depth: ⭐⭐⭐⭐ (Physical model derivation is rigorous)
- Experimental Thoroughness: ⭐⭐⭐ (Dataset scale is limited; few competing baselines)
- Value: ⭐⭐⭐⭐ (Addresses a practical need in underwater robotic vision)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [\[ICCV 2025\] LayerLock: Non-collapsing Representation Learning with Progressive Freezing](layerlock_non-collapsing_representation_learning_with_progressive_freezing.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization](memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)

<!-- RELATED:END -->
