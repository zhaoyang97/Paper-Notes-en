---
title: >-
  [Paper Note] PhysGen3D: Crafting a Miniature Interactive World from a Single Image
description: >-
  [CVPR 2025][3D Vision][Single-image 3D Reconstruction] This paper proposes the PhysGen3D framework, which transforms a single image into a camera-centric interactive 3D scene. By combining the geometric/semantic understanding of visual foundation models with physics-based simulation and rendering, it generates videos that are more physically realistic and controllable than those from commercial I2V models.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Single-image 3D Reconstruction"
  - "Physical Simulation"
  - "Physics-based Rendering"
  - "Interactive Scenes"
  - "Image-to-Video"
date: 2026-05-08
content_hash: 748e05ae5708f81c
---

# PhysGen3D: Crafting a Miniature Interactive World from a Single Image

**Conference**: CVPR 2025  
**arXiv**: [2503.20746](https://arxiv.org/abs/2503.20746)  
**Code**: None (Project Page: [https://by-luckk.github.io/PhysGen3D](https://by-luckk.github.io/PhysGen3D))  
**Area**: 3D Vision / Physical Simulation / Image-to-Video  
**Keywords**: Single-image 3D Reconstruction, Physical Simulation, Physics-based Rendering, Interactive Scenes, Image-to-Video

## TL;DR

This paper proposes the PhysGen3D framework, which transforms a single image into a camera-centric interactive 3D scene. By combining the geometric/semantic understanding of visual foundation models with physics-based simulation and rendering, it generates videos that are more physically realistic and controllable than those from commercial I2V models.

## Background & Motivation

**Background**: Image-to-video (I2V) generation has made significant progress in recent years. Commercial models like Pika, Kling, and Gen-3 can generate realistic videos from a single image. However, these methods essentially learn pixel-level dynamics in the image space implicitly, lacking a physical foundation. On the other hand, physical simulation methods can produce realistic interactive effects but typically require dense data such as multi-view scans or depth sensors.

**Limitations of Prior Work**: (1) Data-driven I2V methods cannot guarantee physical correctness, often generating hallucinated movements that violate physical laws (e.g., objects passing through the ground, unnatural collisions); (2) Users cannot precisely control physical parameters (e.g., velocity, material properties); (3) Existing physics-based methods either require multi-view inputs or are restricted to specific object types (e.g., waterfalls) or 2D domains.

**Key Challenge**: Physical realism requires explicit 3D understanding and physical simulation, but acquiring a complete 3D scene understanding from a single image is a highly ill-posed problem; meanwhile, photorealistic rendering must be achieved while maintaining physical correctness.

**Goal**: To construct a general, controllable, and physically plausible interactive 3D miniature world from a single image, capable of generating physically correct videos based on user-specified initial conditions (e.g., velocity, material properties).

**Key Insight**: Instead of training new models, this work combines and utilizes multiple pre-trained visual foundation models (e.g., GPT-4o for semantic recognition, Grounded-SAM for segmentation, InstantMesh for 3D reconstruction, Dust3r for depth estimation) to construct a "digital twin" pipeline.

**Core Idea**: To construct a complete perception-simulation-rendering pipeline: inferring 3D shape/pose/material/illumination from a single image using visual foundation models $\rightarrow$ simulating dynamics with an MPM physics engine $\rightarrow$ synthesizing the final video through physics-based rendering. The entire process requires no task-specific training.

## Method

### Overall Architecture

The input consists of a single image and user-specified motion/material parameters. The pipeline comprises three stages: (1) **3D World Creation**: segmenting foreground objects, reconstructing 3D meshes, estimating background depth/geometry, calculating 6DoF object poses and scales, and estimating PBR material parameters and physical properties; (2) **Dynamics Simulation**: converting 3D assets to particle representations and performing physical simulation using Taichi Elements (an MPM method); (3) **Physics-based Rendering**: performing PBR rendering with Mitsuba3 and synthesizing foreground dynamics and background via a shadow catcher.

### Key Designs

1. **3D World Reconstruction Based on Visual Foundation Models**:

    - **Function**: Inferring complete 3D scene information from a single image required to support physical simulation.
    - **Mechanism**: Executed step-by-step: (1) using GPT-4o to identify foreground object categories $\rightarrow$ Grounded-SAM to segment instances (iteratively inpainting in case of multi-object occlusion); (2) using InstantMesh (based on Zero123++) to generate multi-view images from segmented object images and reconstruct 3D meshes; (3) using Dust3r to estimate image depth and back-project it into 3D point clouds $\rightarrow$ bilateral normal integration to generate smooth collision surfaces; (4) object pose estimation: employing SuperGlue for 2D-3D feature matching + PnP to solve 6DoF poses in the coarse stage, and using differentiable rendering in the fine stage to jointly optimize Dice loss (mask alignment) and depth consistency loss; (5) using Mitsuba3 inverse rendering to optimize PBR materials (albedo, roughness, metallic), and DiffusionLight to estimate environment illumination; (6) using GPT-4o to query physical properties such as elasticity, density, and friction coefficient, and to estimate a scale factor $k$.
    - **Design Motivation**: Single-image 3D understanding is highly ill-posed, but combining various pre-trained models can complementarily cover geometry, semantics, materials, and other aspects. The step-by-step pipeline ensures that each step is handled by the state-of-the-art model in that specific domain.

2. **Dynamics Simulation Based on MPM**:

    - **Function**: Simulating the dynamic behavior of various materials based on physical parameters and user-specified initial conditions.
    - **Mechanism**: Converting 3D meshes into particle representations (outlier removal $\rightarrow$ internal filling $\rightarrow$ voxel downsampling) and using the MPM engine of Taichi Elements for simulation. It supports various materials such as rigid bodies, soft bodies, and granular materials. A key technique is using the scale factor $k$ (the ratio of the object's real-world size to its scale in the scene) to scale the physical parameters rather than the assets themselves—for example, the acceleration of gravity becomes $k \times 9.8$—thereby avoiding the impact of scaling on numerical simulation stability. Users can control the simulation effects by setting different initial velocities and material types (rigid/soft/granular).
    - **Design Motivation**: Compared to pure particle methods, the MPM method is more stable when handling collisions and multi-material coupling, balancing accuracy and efficiency through a hybrid point-voxel representation. The dimensionless treatment via the scale factor addresses the issue of unrealistic physical simulation caused by insufficient depth estimation accuracy.

3. **Physics-based Final Rendering**:

    - **Function**: Rendering simulation results into realistic videos with correct lighting and shadows.
    - **Mechanism**: Simulated particle trajectories are used to calculate vertex motion via motion interpolation, driving mesh deformation. PBR rendering is performed using Mitsuba3 with optimized material parameters. Instead of incorporating the entire background into the rendering pipeline (which would be too complex), a 3D shadow catcher is constructed from the background depth map. Two-pass shadow mapping is employed to extract shadows and global illumination effects, respectively. Finally, the foreground object and shadows are composted onto the pre-inpainted (via LaMA) background.
    - **Design Motivation**: Direct full-scene rendering is computationally expensive, and accurate background texture reconstruction is challenging. The shadow catcher method renders only the dynamic effects (shadows, indirect illumination) and composites them with the inpainted static background, which balances efficiency and visual realism.

### Loss & Training

The entire pipeline is training-free, using differentiable rendering to iteratively optimize PBR parameters only during the appearance optimization step. The loss used for pose estimation is $\mathcal{L} = \mathcal{L}_{dice} + \mathcal{L}_{depth}$. The albedo is optimized using a constrained tone-mapping function $y(x) = ax^3 + bx^2 + cx + d$ with $y(0)=0, y(1)=1$.

## Key Experimental Results

### Main Results (User Evaluation, 5-point Likert Scale)

| Method | Physical Realism↑ | Visual Quality↑ | Semantic Consistency↑ |
|------|-----------|----------|-----------|
| Kling 1.0 | 2.811 | 3.566 | 2.467 |
| Gen-3 | 2.283 | 3.582 | 1.886 |
| Pika 1.5 | 2.412 | 3.314 | 2.016 |
| **PhysGen3D** | **3.707** | 3.411 | **3.866** |

### GPT-4o Automatic Evaluation

| Method | Physical Realism↑ | Visual Quality↑ | Semantic Consistency↑ |
|------|-----------|----------|-----------|
| Kling 1.0 | 0.563 | 0.874 | 0.596 |
| DragAnything | 0.645 | 0.756 | 0.380 |
| **PhysGen3D** | **0.752** | 0.867 | **0.796** |

### Key Findings

- PhysGen3D significantly outperforms all commercial I2V models in physical realism and semantic consistency, though its visual quality is slightly lower (as the non-learning-based method lacks enhancements like super-resolution).
- Even with carefully tuned prompts and motion brush controls, commercial I2V models still frequently generate motions that violate physical laws.
- The automatic evaluation results of GPT-4o are highly consistent with human evaluation, validating the reliability of GPT-4o as an evaluation tool.
- Ablation studies demonstrate that both pose optimization and inverse texture optimization are necessary; without pose optimization, positions are inaccurate, and without texture optimization, color tones do not match.
- The framework supports generating multiple dynamic effects from the same image (by changing materials or velocity directions), demonstrating high controllability.

## Highlights & Insights

- Highly clear methodology: A three-stage "perception $\rightarrow$ simulation $\rightarrow$ rendering" pipeline, using the most suitable tool for each stage.
- Truly achieves "understanding-based rather than learning-based" video generation, inherently guaranteeing physical correctness.
- The training-free design allows the framework to generalize immediately to arbitrary object categories.
- Users can precisely control physical parameters (velocity vectors, material types, elasticity coefficients, etc.), offering interactivity unparalleled by commercial I2V models.
- Supports additional functions such as video editing (object replacement/deletion) and dense 3D tracking.

## Limitations & Future Work

- Designed for object-centric scenes (one to several objects) and cannot handle complex multi-object scenes or heavy occlusions.
- Rendering quality may exhibit artifacts under extreme lighting and heavy shadow conditions.
- Insufficient recovery of detailed structures (such as teapot spouts) in single-image 3D reconstruction leads to simulation failures.
- The MPM method has limited accuracy when handling thin structures.
- Future work could integrate I2V models as a post-processing step to improve visual quality, or introduce differentiable simulation for end-to-end optimization.

## Related Work & Insights

- Compared to Liu et al. (2D rigid body simulation), PhysGen3D extends to 3D space and multiple materials.
- The methodology of "combining foundation models to construct digital twins" can be generalized to fields such as robotic manipulation and scene understanding.
- Physical simulation versus data-driven video generation are two complementary directions, and combining them represents a future trend.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Overall Rating | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] GenVDM: Generating Vector Displacement Maps From a Single Image](genvdm_generating_vector_displacement_maps_from_a_single_image.md)
- [\[CVPR 2025\] Floating No More: Object-Ground Reconstruction from a Single Image](floating_no_more_object-ground_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)
- [\[CVPR 2025\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)

</div>

<!-- RELATED:END -->
