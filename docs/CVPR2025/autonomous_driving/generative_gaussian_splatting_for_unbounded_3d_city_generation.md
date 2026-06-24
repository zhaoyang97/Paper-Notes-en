---
title: >-
  [Paper Note] Generative Gaussian Splatting for Unbounded 3D City Generation
description: >-
  [CVPR 2025][Autonomous Driving][3D Gaussian Splatting] Proposes GaussianCity, the first framework to apply 3D Gaussian Splatting to unbounded 3D city generation. By introducing a compact intermediate representation called BEV-Point, GPU memory consumption is decoupled from the scene scale (remaining constant). Additionally, a Point Serializer is designed to convert unordered BEV points into ordered sequences to capture structural and contextual features. This achieves state-o…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "3D Gaussian Splatting"
  - "unbounded city generation"
  - "BEV-Point representation"
  - "efficient rendering"
  - "scene synthesis"
date: 2026-05-08
content_hash: f0e99be5a46c135c
---

# Generative Gaussian Splatting for Unbounded 3D City Generation

**Conference**: CVPR 2025  
**arXiv**: [2406.06526](https://arxiv.org/abs/2406.06526)  
**Code**: [https://haozhexie.com/project/gaussian-city](https://haozhexie.com/project/gaussian-city)  
**Area**: Autonomous Driving  
**Keywords**: 3D Gaussian Splatting, unbounded city generation, BEV-Point representation, efficient rendering, scene synthesis

## TL;DR

Proposes GaussianCity, the first framework to apply 3D Gaussian Splatting to unbounded 3D city generation. By introducing a compact intermediate representation called BEV-Point, GPU memory consumption is decoupled from the scene scale (remaining constant). Additionally, a Point Serializer is designed to convert unordered BEV points into ordered sequences to capture structural and contextual features. This achieves state-of-the-art (SOTA) performance in both drone and street-view city generation, with rendering speeds 60 times faster than CityDreamer (which is based on NeRF).

## Background & Motivation

**Background**: 3D city generation is one of the most challenging tasks in 3D content creation, with wide applications in gaming, animation, film, and VR. InfiniCity and CityDreamer use NeRF to achieve unbounded city generation, yielding promising results. Over the past year, 3D Gaussian Splatting (3D-GS) has received widespread attention in 3D generation due to its efficient GPU rasterization rendering and flexible detail representation capabilities.

**Limitations of Prior Work**: NeRF-based methods sample all points with the same density and aggregate them along the rays, leading to low inference efficiency and lost details. Conversely, existing 3D-GS generative models can only handle objects or scenes of limited scale. When the scene expands to the city scale, the number of Gaussian points must swell to billions. A 10km² city scene could require hundreds of GBs of GPU memory, making the direct application of 3D-GS to unbounded city generation entirely infeasible.

**Key Challenge**: The fundamental conflict between the efficient rendering advantage of 3D-GS and the explosive growth of GPU memory and storage in large-scale scenes.

**Goal**: How to preserve the efficient rendering of 3D-GS while preventing GPU memory consumption from growing with the scene scale, thereby achieving unbounded 3D city generation?

**Key Insight**: The authors observe that with fixed camera parameters, no matter how large the scene is, the number of visible BEV points in each frame remains constant. Therefore, Gaussian attributes can be decoupled into position-dependent parts (encoded into BEV maps) and style-dependent parts (encoded into lookup tables), performing rendering and optimization only on visible points.

**Core Idea**: Use BEV-Point as a compact intermediate representation, retaining only visible points and recovering spatial structure through a Point Serializer, to achieve unbounded city Gaussian generation with constant GPU memory.

## Method

### Overall Architecture

The pipeline of GaussianCity consists of four steps: (1) generating BEV points from local patches of BEV maps (height map $\mathbf{H}$, semantic map $\mathbf{S}$, density map $\mathbf{D}$) and filtering visible points; (2) generating BEV-Point attributes (instance labels, coordinates, scene features) for each point and a style lookup table for each instance; (3) the BEV-Point Decoder generates Gaussian attributes through a Point Serializer + Point Transformer + Modulated MLP; (4) the Gaussian Rasterizer renders the final image.

### Key Designs

1. **BEV-Point Compact Representation**:

    - Function: Decouple and compress Gaussian attributes, making GPU memory consumption independent of scene scale.
    - Mechanism: Through the semantic and height maps of the BEV map, pixels are stretched along the height direction to generate a 3D point set $\mathbf{C}_F$. A binary density map $\mathbf{D}$ is introduced to adaptively adjust the sampling density for different semantic categories (reducing density for simple textures like road surfaces and increasing density for complex textures like building facades). The key step is to obtain the visibility map $\mathcal{V}$ through ray intersection, keeping only visible points $\mathbf{C}_A$. Because the number of visible points is constant under fixed camera parameters, GPU memory consumption does not grow as the scene expands.
    - Design Motivation: Compared to directly using 3D-GS (where GPU memory grows linearly with the number of points), BEV-Point compresses ~20 million BEV points across the full scene to ~hundreds of thousands of visible points. Additionally, file storage is compressed from the attribute dimension (~60 dimensions per point) to the form of BEV maps + lookup tables.

2. **Point Serializer**:

    - Function: Convert unordered BEV points into an ordered sequence with spatial locality.
    - Mechanism: Design a linearization function $\mathcal{L}(x,y,z,g) = \lfloor x/g^2 + y/g + z \rfloor$ to map each point's 3D coordinates to an integer sequence number. The discretization granularity is controlled by grid size $g$. After sorting, points adjacent in the data structure are also adjacent in space, enabling the subsequent Transformer to effectively capture contextual relationships within local windows.
    - Design Motivation: Points sampled along rays in NeRF naturally maintain spatial correlation, but BEV points are unordered point clouds, and processing them directly with an MLP would lose spatial structure. The Serializer restores this spatial locality through the concept of space-filling curves.

3. **Style Lookup Table and Modulated MLP**:

    - Function: Control the appearance diversity of different instances at low cost.
    - Mechanism: Learn a style vector $\mathbf{z}_T^i \sim \mathcal{N}(0,1)$ for each instance (buildings, vehicles, etc.) and store it in a lookup table $\mathcal{T}$. When generating Gaussian attributes, the final properties (color, scale, rotation, etc.) are generated by modulating BEV point features with the style vector through a Modulated MLP. Consequently, all BEV points of the same building share a single style code, significantly reducing the parameter count.
    - Design Motivation: The appearance diversity of buildings and vehicles in urban scenes requires instance-level control, but storing independent attributes for each point in a global 3D-GS is too expensive. The style table compresses high-dimensional attributes into low-dimensional style codes + a prediction network.

### Loss & Training

A hybrid loss is used: $\ell = \lambda_{L1} \|\hat{\mathbf{R}} - \mathbf{R}\| + \lambda_{VGG} \text{VGG}(\hat{\mathbf{R}}, \mathbf{R}) + \lambda_{GAN} \text{GAN}(\hat{\mathbf{R}}, \mathbf{S}_G)$, where the L1 loss ensures pixel-level reconstruction, the VGG perceptual loss ensures semantic consistency, and the GAN adversarial loss (conditioned on the semantic map) ensures realism.

## Key Experimental Results

### Main Results

Comparison of urban generation quality on GoogleEarth and KITTI-360 datasets:

| Method | FID↓ | KID↓ | Rendering Speed (FPS) | Dataset |
|------|------|------|---------------|--------|
| CityDreamer | 66.05 | 3.06 | 0.18 | GoogleEarth (Drone) |
| **Ours** | **57.92** | **2.58** | **10.72** | GoogleEarth (Drone) |
| CityDreamer | 78.41 | 6.35 | 0.17 | KITTI-360 (Street) |
| **Ours** | **69.23** | **5.14** | **9.89** | KITTI-360 (Street) |

GaussianCity comprehensively outperforms CityDreamer on FID/KID metrics and is **60 times** faster in rendering speed (10.72 vs 0.18 FPS).

### Ablation Study

| Configuration | FID↓ | KID↓ | Description |
|------|------|------|------|
| Full model | 57.92 | 2.58 | Full model |
| w/o Point Serializer | 63.18 | 3.12 | Without serialization, quality drops significantly |
| w/o Density Map $\mathbf{D}$ | 60.45 | 2.89 | Oversampling in simple regions |
| w/o Style Lookup Table | 62.37 | 3.04 | Reduced instance diversity |
| w/o Scene Feature $\mathbf{F}_S$ | 61.93 | 2.96 | Missing contextual information |

### Key Findings

- Point Serializer contributes the most to generation quality (the FID increases by 5+ without it), indicating that structuring unordered point clouds is crucial for generative tasks.
- The adaptive sampling strategy of the density map $\mathbf{D}$ effectively balances quality and efficiency—sampling less in simple texture areas (road surfaces) and densely in complex areas (building facades).
- The BEV-Point representation truly achieves "constant" GPU memory—when the scene expands from 1km² to 10km², the GPU memory only goes from 3.2GB to 3.3GB, while directly using 3D-GS causes GPU memory to grow from 12GB to over 120GB.

## Highlights & Insights

- **Constant GPU Memory Unbounded Scene Generation**: Through the simple insight of "looking only at visible points," GPU memory is shifted from growing linearly with the scene to remaining constant. This perspective can be transferred to any large-scale scene task based on point representations.
- **Spatial Locality Recovery of Point Serializer**: Using a 1D sorting function to recover 3D spatial locality allows standard Transformers to process unordered point clouds directly, which is simpler and more efficient than complex graph attention or point cloud networks.
- The practical significance of the 60x acceleration is immense—shifting from 0.18 FPS (unusable for interaction) to 10.72 FPS (near real-time), which makes interactive applications of 3D city generation possible for the first time.

## Limitations & Future Work

- Currently relies on predefined BEV maps (semantic map, height map) as inputs, lacking the capability to generate layouts from scratch. Combining this with layout generation models can be considered.
- The internal structures of buildings are completely invisible, rendering only the outer shells, which may result in visual glitches (clipping/breaking) when observed up close in street-view mode.
- The Point Serializer uses a simple linear mapping, which might be suboptimal for space-filling of complex topologies (e.g., bridges, viaducts). Z-order curves or Hilbert curves could be explored.
- No modeling of dynamic objects (pedestrian or vehicle motion), meaning only static cities can be generated at present.

## Related Work & Insights

- **vs CityDreamer**: CityDreamer uses NeRF to achieve unbounded city generation but renders extremely slowly. GaussianCity uses 3D-GS to gain a 60x speedup with better quality. The core difference is that BEV-Point resolves the GPU memory bottleneck of 3D-GS.
- **vs InfiniCity**: InfiniCity is also based on NeRF and achieves unbounded generation through patch-based generation, but block boundaries may suffer from discontinuities. GaussianCity's BEV-Point naturally supports cross-block consistency.
- **vs LGM/GS-LRM**: These 3D-GS generation methods are only applicable at the single-object level, whereas GaussianCity extends this to the city level for the first time.

## Rating

- Novelty: ⭐⭐⭐⭐ The BEV-Point decoupling concept is novel and holds significant engineering value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive analyses on GPU memory, efficiency comparisons, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Exquisite illustrations with a clear presentation of methods.
- Value: ⭐⭐⭐⭐⭐ The 60x speedup makes city-scale 3D generation practical for the first time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](../../ICCV2025/autonomous_driving/3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)
- [\[CVPR 2025\] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [\[CVPR 2025\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [\[CVPR 2025\] PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting](pansplat_4k_panorama_synthesis_with_feed-forward_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
