---
title: >-
  [Paper Note] RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] RainyGS integrates physically-based raindrop simulation and shallow water dynamics with the 3D Gaussian Splatting rendering framework. It achieves high-fidelity, physically accurate, and real-time (>30fps) dynamic rain effect synthesis in open-world scenes for the first time, supporting flexible control from light rain to downpours.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Rain Effect Synthesis"
  - "Physical Simulation"
  - "Shallow Water Equations"
  - "Screen Space Ray Tracing"
date: 2026-05-08
content_hash: 5b25d48d1d8bcee3
---

# RainyGS: Efficient Rain Synthesis with Physically-Based Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2503.21442](https://arxiv.org/abs/2503.21442)  
**Code**: [https://pku-vcl-geometry.github.io/RainyGS/](https://pku-vcl-geometry.github.io/RainyGS/)  
**Area**: 3D Vision / Scene Editing  
**Keywords**: 3D Gaussian Splatting, Rain Effect Synthesis, Physical Simulation, Shallow Water Equations, Screen Space Ray Tracing

## TL;DR

RainyGS integrates physically-based raindrop simulation and shallow water dynamics with the 3D Gaussian Splatting rendering framework. It achieves high-fidelity, physically accurate, and real-time (>30fps) dynamic rain effect synthesis in open-world scenes for the first time, supporting flexible control from light rain to downpours.

## Background & Motivation

**Background**: 3D scene modeling has made significant progress in recent years, with NeRF and 3DGS becoming powerful scene representation tools. Meanwhile, physically-based weather simulation can produce realistic effects but relies on manual scene construction. The two have complementary advantages but remain difficult to combine.

**Limitations of Prior Work**: (1) Video generation methods (e.g., Runway) can generate visually appealing rain styles but lack 3D consistency, physical accuracy, and dynamic evolution; (2) ClimateNeRF only supports static weather effects; (3) Gaussian Splashing uses inefficient particle simulations and cannot run in real time; (4) Data-driven rain synthesis methods (such as CycleGAN) only process single images and cannot guarantee multi-view consistency.

**Key Challenge**: Rain scenes involve extremely complex physical and rendering phenomena—raindrop streaks, ground puddles, ripples, reflections, refractions, and splashes must appear simultaneously and intensities must evolve over time, requiring highly efficient computation. Existing methods either lack physical accuracy, are too computationally expensive, or lack 3D consistency.

**Goal**: To implement a complete physical rain effect system within the 3DGS framework that covers all key rainy weather phenomena while achieving real-time performance.

**Key Insight**: The authors note that water in rain scenes can be approximated as a height field (shallow water assumption), which is naturally compatible with the 3DGS rasterization pipeline. Combining this with screen-space ray tracing for reflection rendering avoids the massive computational overhead of performing 3D ray tracing on 3DGS entities.

**Core Idea**: Simulate water dynamics on a height map using shallow water equations and implement reflections/refractions in the rasterization pipeline via screen-space ray tracing, seamlessly integrating both with 3DGS.

## Method

### Overall Architecture

Given multi-view image inputs, the scene's appearance and geometry are first reconstructed using PGSR. Then, auxiliary maps (height, depth, normal maps) are extracted. Shallow water simulation runs on the height map to generate puddle dynamics. Finally, raindrop streaks, reflections, refractions, and splashes are synthesized using a reflection-aware water surface rasterization method, which are combined to produce the final dynamic rainy images.

### Key Designs

1. **Heightfield-Based Shallow Water Simulation**:

    - **Function**: Simulates the dynamic behavior of ground water accumulation (waves, ripples, raindrop splashing).
    - **Mechanism**: Solves the shallow water equations (SWE) $\partial h/\partial t + (\mathbf{u}\cdot\nabla)h = -h(\nabla\cdot\mathbf{u})$ on a 2D height map. Raindrops are randomly generated, and their volumes are added to the height field $h_{i,j} \leftarrow h_{i,j} + 4\pi r_i^3/(3\Delta x^2)$ after collision detection. A two-layer height map (ground and obstacles) is maintained along with a semi-Lagrangian advection scheme. The key is to maintain the water surface height field $\eta = H + h$, where $H$ is the ground terrain height.
    - **Design Motivation**: The shallow water assumption is highly suitable for rain—accumulated water is indeed shallow, and the height field representation circumvents the internal geometry missing issue in 3DGS (since GS has no internal structure). This dramatically improves efficiency compared to particle simulations.

2. **Reflection-Aware Water Rasterization**:

    - **Function**: Implements physically accurate reflections, refractions, and specular highlights within the rasterization pipeline.
    - **Mechanism**: Executed in two rendering passes. The first pass renders the water surface layer $I_0 = (1-F)I_{\text{refra}} + F(I_{\text{spec}} + I_{\text{highl}})$, where $F$ is the Fresnel coefficient. Specular reflection $I_{\text{spec}}$ is obtained via screen-space ray tracing (SSR) marching on the already rasterized image—ray-marching along the reflection direction only in the 2D image, bypassing true 3D ray tracing. Refraction is approximated using image warping $I_{\text{refra}}(u,v) = I_{\text{src}}(u+n_uk, v+n_vk)$. Highlights are computed via the Blinn-Phong model. The second pass rasterizes rain streaks and splashes.
    - **Design Motivation**: 3D ray tracing is extremely expensive in GS scenes (requiring traversal of a massive number of Gaussians). Screen-space ray tracing operates solely at the 2D level, integrating perfectly with the rasterization pipeline. Since reflections contribute most to visual realism, they are the focus of rendering budget allocation.

3. **Accurate Geometry Reconstruction based on PGSR**:

    - **Function**: Provides high-quality scene geometry required for rain simulation.
    - **Mechanism**: Uses PGSR (Gaussian with planar constraints) instead of standard 3DGS for scene modeling, incorporating a monocular normal estimation prior to supervise the normal maps. PCA is used to identify the ground plane direction, and a height map is obtained by rendering depth from an orthographic camera. Auxiliary maps (depth, normals, appearance) are extracted via alpha blending.
    - **Design Motivation**: Rain simulation demands high geometry accuracy—ground flatness directly dictates water accumulation behavior, and normal accuracy guides the reflection direction. Standard 3DGS does not provide sufficient geometry reconstruction quality.

### Loss & Training

The scene modeling stage uses PGSR's standard training pipeline plus a normal prior loss. The rain effect synthesis stage requires no training; it is a pure physical simulation + rendering process at inference time. Users can control rain levels via parameters such as raindrop density, velocity, and orientation.

## Key Experimental Results

### Main Results

Evaluated on MipNeRF360 (Garden, Treehill, Bicycle) and Tanks and Temples (Family, Truck) datasets.

Performance Comparison:

| Method | Frame Rendering Time | Peak VRAM |
|------|-----------|---------|
| PGSR (Without Rain) | 0.007s | 7.989 GB |
| PGSR + RT (3D Ray Tracing) | 1.942s | 14.161 GB |
| Runway-V2V | ~0.4s | NA |
| RainyGS | **0.032s** | 8.561 GB |

### Ablation Study

Fig.3 illustrates the progressive stacking of rendering modules:

| Configuration | Effect Description |
|------|---------|
| $I_{\text{src}}$ | Original Scene |
| + $I_{\text{spec}}$ | Specular reflection is added, creating ground reflections |
| + $I_{\text{refra}}$ | Refraction is added, introducing warping effects on the ground |
| + $I_{\text{highl}}$ | Specular highlights are added, introducing solar glare on the water surface |
| + Fresnel Grading | The overall scene tone becomes darker and more realistic |
| + Rain Streaks | Full rain effect |

### Key Findings

- RainyGS only adds 0.025s/frame and 0.572 GB VRAM compared to PGSR, whereas the 3D ray tracing solution adds nearly 2s/frame—making screen-space ray tracing highly efficient.
- Runway-V2V can generate rainy styles visually but lacks 3D consistency (rain effects are inconsistent across different views of the same scene) and is physically inaccurate (lacks ripple responses for water waves).
- Rain Motion fails to generate water accumulation effects and can only overlay simple rain streaks.
- Instruct-GS2GS only performs static edits and cannot generate dynamic rain effects.
- The pipeline also works on autonomous driving (Waymo) scenes, showcasing the scalability of the framework to large-scale environments.

## Highlights & Insights

- **Clever Choice of Shallow Water Equations**: Rain puddles naturally satisfy the "shallow water" assumption. This physical modeling choice is both accurate and efficient. Furthermore, the height field representation sidesteps the lack of internal geometry in 3DGS, killing two birds with one stone.
- **Engineering Wisdom of Screen-Space Ray Tracing**: Reducing reflection calculation from 3D space to 2D screen space integrates seamlessly into the rasterization pipeline. Although it cannot trace reflections of occluded objects, the reflections for visible regions are realistic enough.
- **Complete Effect System**: Covering all key visual elements of rainy scenes—from rain streaks, puddle accumulation, and water waves to reflections, refractions, and Fresnel tone adjustment—rather than focusing on a single effect.

## Limitations & Future Work

- The shallow water model is not applicable to deep water scenes (like floods) and complex multi-layered water bodies.
- Screen-space rendering cannot reflect occluded objects.
- Geometry accuracy depends heavily on scene reconstruction quality—insufficient input views may lead to uneven ground geometry, causing unnatural puddle distribution.
- Future extensions could include fog, snow, and other weather conditions to build a unified weather simulation framework.
- By combining this with differentiable rendering, rain effects could serve as data augmentation to improve the robustness of autonomous driving systems.

## Related Work & Insights

- **vs ClimateNeRF**: ClimateNeRF is a pioneer in this field but only supports static weather (frozen floods, fog, snow). RainyGS achieves dynamically evolving rainy effects.
- **vs Gaussian Splashing**: Gaussian Splashing treats each Gaussian as a physical particle for fluid simulation, which is extremely expensive to compute. RainyGS drastically reduces costs through heightfield abstraction and screen-space rendering.
- **vs PhysGaussian**: PhysGaussian focuses on solid-body physics (such as elastic deformation), whereas RainyGS focuses on fluid-related effects; the two are complementary.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first to implement complete physical rain simulation in 3DGS, with a highly appropriate technical selection of Shallow Water + SSR.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Extensive multi-scene validation and performance analysis are provided, though it lacks quantitative measures of rain realism (such as user studies).
- **Writing Quality**: ⭐⭐⭐⭐ Pipeline is clear, modular descriptions are easy to understand, and illustrations are abundant.
- **Value**: ⭐⭐⭐⭐ High application potential in open-world rain synthesis (autonomous driving, gaming, AR), with real-time performance acting as a key advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[CVPR 2025\] SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis](splatflow_multi-view_rectified_flow_model_for_3d_gaussian_splatting_synthesis.md)
- [\[CVPR 2025\] Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)
- [\[CVPR 2025\] ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [\[CVPR 2025\] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
