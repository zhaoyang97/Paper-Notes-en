---
title: >-
  [Paper Note] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability
description: >-
  [NeurIPS 2025][Autonomous Driving][large-scale scene generation] This paper presents X-Scene, a unified large-scale driving scene generation framework that supports multi-granularity control ranging from high-level text prompts to low-level BEV layouts. By jointly generating 3D semantic occupancy, multi-view images, and videos, and leveraging consistency-aware extrapolation for large-scale scene expansion, X-Scene comprehensively outperforms existing methods in generation qua…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "large-scale scene generation"
  - "multi-granularity control"
  - "occupancy generation"
  - "3DGS reconstruction"
  - "autonomous driving simulation"
date: 2026-05-08
content_hash: 58bc84fb56a51456
---

# X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability

**Conference**: NeurIPS 2025
**arXiv**: [2506.13558](https://arxiv.org/abs/2506.13558)  
**Code**: [https://x-scene.github.io/](https://x-scene.github.io/)  
**Area**: Autonomous Driving / Scene Generation
**Keywords**: large-scale scene generation, multi-granularity control, occupancy generation, 3DGS reconstruction, autonomous driving simulation

## TL;DR

This paper presents X-Scene, a unified large-scale driving scene generation framework that supports multi-granularity control ranging from high-level text prompts to low-level BEV layouts. By jointly generating 3D semantic occupancy, multi-view images, and videos, and leveraging consistency-aware extrapolation for large-scale scene expansion, X-Scene comprehensively outperforms existing methods in generation quality (FID 11.29) and downstream tasks.

## Background & Motivation

Diffusion models have demonstrated remarkable success in autonomous driving data synthesis and simulation. However, existing works primarily focus on **temporally consistent video generation** (e.g., MagicDrive, DriveDreamer), while **spatially consistent large-scale 3D scene generation** remains an underexplored direction.

Core limitations of prior work:

**SemCity**: Can generate city-level 3D occupancy grids but lacks appearance details, making it unsuitable for realistic simulation.

**UniScene / InfiniCube**: Jointly generate occupancy and images but require manually designed large-scale layouts as input, resulting in complex pipelines with limited flexibility.

**General large-scale urban generation methods** (InfiniCity, CityDreamer): Not tailored for driving scenarios, lacking precise road layouts and dynamic objects.

Large-scale driving scene generation faces three core challenges: flexible controllability, high-fidelity geometry and appearance, and large-scale consistency.

**Core Idea**: Construct a unified cascaded generation pipeline from text/layout to occupancy–image–video, enable large-scale scene expansion via consistency-aware extrapolation, and reconstruct scenes as 3DGS to support downstream applications.

## Method

### Overall Architecture

X-Scene consists of three core modules: (1) multi-granularity controllability—combining high-level text and low-level layout conditions; (2) joint occupancy–image–video generation—ensuring cross-modal alignment and temporal consistency; (3) large-scale scene extrapolation and 3DGS reconstruction.

### Key Designs

1. **Multi-Granularity Controllability**:

    - **Function**: Supports multi-level scene control ranging from coarse text prompts to fine-grained geometric layouts.
    - **Mechanism**: The high-level path employs a RAG-augmented LLM to generate detailed scene descriptions and construct scene graphs, followed by graph convolution and conditional diffusion for layout generation. The low-level path directly utilizes BEV layouts and 3D bounding boxes.
    - **Design Motivation**: High-level control is suitable for rapid prototyping, while low-level control is suited for precise simulation; the two paths are complementary.

2. **Joint Occupancy–Image–Video Generation**:

    - **Function**: Generates aligned occupancy fields, multi-view images, and sequential videos following a 3D-to-2D hierarchical order.
    - **Occupancy Generation**: Triplane representation with a proposed triplane deformable attention mechanism to mitigate information loss caused by downsampling.
    - **Image Generation**: Occupancy voxels are converted to 3D Gaussians to render semantic/depth maps, which are fused into geometric embeddings to condition the image diffusion model.
    - **Video Generation**: Preceding images serve as reference frames; only temporal attention layers are fine-tuned, enabling autoregressive streaming generation.
    - **Design Motivation**: The 3D-to-2D hierarchical generation ensures geometry–appearance consistency.

3. **Large-Scale Scene Extrapolation and 3DGS Reconstruction**:

    - **Function**: Extends local generation to large-scale coherent environments.
    - **Occupancy Extrapolation**: The triplane is decomposed into three 2D planes for extrapolation, with overlapping masks for synchronized denoising.
    - **Image Extrapolation**: The diffusion model is fine-tuned to condition on reference images and camera embeddings for novel view synthesis.
    - **Design Motivation**: Consistency-aware extrapolation ensures structural coherence in overlapping regions.

### Loss & Training

The three diffusion models are trained independently, all using a noise prediction objective. For video diffusion, only the temporal attention layers are fine-tuned.

## Key Experimental Results

### Main Results

Occupancy generation:

| Method | FID3D | F3D | P3D | R3D |
|------|-------|------|------|------|
| UniScene | 529.6 | 0.396 | 0.382 | 0.412 |
| **X-Scene** | **258.8** | **0.778** | **0.769** | **0.787** |

Multi-view image generation:

| Method | FID | Road mIoU | Veh. mIoU | mAP | NDS |
|------|------|-----------|-----------|------|------|
| MagicDrive | 16.20 | 61.05 | 27.01 | 12.30 | 23.32 |
| DreamForge | 14.61 | 65.27 | 28.36 | 13.01 | 22.16 |
| **X-Scene (224×400)** | **11.29** | 66.48 | 29.76 | 16.28 | 26.26 |
| **X-Scene (448×800)** | 12.77 | **69.06** | **33.27** | **27.65** | **34.48** |

Data augmentation effect:

| Data | 3D mAP | BEV Road mIoU | BEV Veh. mIoU |
|------|--------|-------------|-------------|
| Real only | 34.5 | 74.30 | 36.00 |
| +UniScene | 36.5 | 81.69 | 41.62 |
| **+X-Scene** | **39.9** | **83.37** | **43.05** |

### Ablation Study

| Configuration | IoU | mIoU | FID3D | F3D | Notes |
|------|------|-------|--------|------|------|
| Full | **85.6** | **92.4** | **258.8** | **0.778** | — |
| w/o Deform Attn (50×50) | 64.7 | 74.2 | 462.4 | 0.510 | Severe downsampling loss |
| w/ Deform Attn (50×50) | 66.6 | 76.6 | 436.1 | 0.522 | +2.4% with deformable attention |
| w/o Layout Cond | 85.6 | 92.4 | 1584 | 0.237 | FID increases by 6× |

### Key Findings

- Triplane-VAE achieves reconstruction mIoU of 92.4%, substantially surpassing UniScene's 73.7%.
- FID3D is reduced by 51.2% (258.8 vs. 529.6).
- Data augmentation improves 3D detection mAP by 5.4 points (34.5 → 39.9).
- Training with 7 frames outperforms the 16-frame baseline (FVD 179.7 vs. 217.9), validating the efficiency of autoregressive temporal modeling.

## Highlights & Insights

- First end-to-end driving scene generation framework: text → scene graph → layout → occupancy → image → video → 3DGS.
- The dual-path multi-granularity control design is highly practical.
- Triplane deformable attention substantially improves reconstruction accuracy while maintaining encoding efficiency.
- Consistency-aware extrapolation elegantly extends local generation to large-scale scenes.

## Limitations & Future Work

- Progressive extrapolation may accumulate errors in extremely large-scale scenes.
- The scene-graph-to-layout diffusion generation depends on the diversity of training data.
- Spatiotemporally consistent generation of dynamic objects remains unexplored.

## Related Work & Insights

- Complementary to MagicDrive and UniScene: X-Scene emphasizes spatial scalability and controllability.
- The RAG + scene graph + diffusion pipeline for text-to-layout generation is transferable to domains such as indoor scene generation.
- The occupancy-first generation paradigm offers a new framework for 3D perception data augmentation.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)
- [\[CVPR 2026\] Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals](../../CVPR2026/autonomous_driving/rascene_high-fidelity_3d_scene_imaging_with_mmwave_communication_signals.md)
- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[ICCV 2025\] Decoupled Diffusion Sparks Adaptive Scene Generation](../../ICCV2025/autonomous_driving/decoupled_diffusion_sparks_adaptive_scene_generation.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](../../ICCV2025/autonomous_driving/controllable_3d_outdoor_scene_generation_via_scene_graphs.md)

</div>

<!-- RELATED:END -->
