---
title: >-
  [Paper Note] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation
description: >-
  [CVPR 2025][3D Vision][Sim2Real] Vid2Sim proposes a real2sim framework that converts monocular videos into realistic and interactive simulation environments. By employing geometrically consistent Gaussian Splatting reconstruction and a hybrid scene representation (GS+Mesh), it supports the reinforcement learning training of urban navigation agents, improving success rates by 31.2% in digital twins and 68.3% in the real world.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Sim2Real"
  - "3D Gaussian Splatting"
  - "Scene Reconstruction"
  - "Visual Navigation"
  - "Hybrid Scene Representation"
date: 2026-05-08
content_hash: 1bf0918ced08047c
---

# Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation

**Conference**: CVPR 2025  
**arXiv**: [2501.06693](https://arxiv.org/abs/2501.06693)  
**Code**: [https://metadriverse.github.io/vid2sim/](https://metadriverse.github.io/vid2sim/)  
**Area**: 3D Vision / Autonomous Driving  
**Keywords**: Sim2Real, 3D Gaussian Splatting, Scene Reconstruction, Visual Navigation, Hybrid Scene Representation

## TL;DR
Vid2Sim proposes a real2sim framework that converts monocular videos into realistic and interactive simulation environments. By employing geometrically consistent Gaussian Splatting reconstruction and a hybrid scene representation (GS+Mesh), it supports the reinforcement learning training of urban navigation agents, improving success rates by 31.2% in digital twins and 68.3% in the real world.

## Background & Motivation

**Background**: Training navigation agents in simulated environments is a mainstream approach in robotics, but the sim2real gap remains a core challenge. Traditional methods alleviate this through domain randomization and system identification, but are limited by the fidelity of the simulators themselves. Neural reconstruction techniques such as NeRF and 3DGS can reconstruct photorealistic 3D scenes from real-world data, but most focus solely on novel view synthesis and do not support physical interaction.

**Limitations of Prior Work**: (1) Traditional simulators (e.g., Habitat, Gibson) have limited visual fidelity, and environment types are constrained by pre-built 3D assets; (2) Although 3DGS produces photorealistic rendering, it lacks physical interaction capabilities and collision detection; (3) Video2Game attempts to extend NeRF reconstruction to gaming scenes, but its visual quality is limited by textured mesh representations, and it is only suitable for games rather than robotics training; (4) Reconstruction from in-the-wild videos suffers from poor geometric quality—3DGS overfits to training views and produces floater artifacts when the exploration viewpoint deviates.

**Key Challenge**: A representation conflict exists between high-quality visual rendering and interactive physical simulation—GS excels at rendering but does not support collision detection, while Mesh supports physics but has limited visual quality.

**Goal**: Construct realistic yet physically interactive simulation environments from monocular videos for the RL training of urban navigation agents, thereby minimizing the sim2real gap.

**Key Insight**: A hybrid scene representation combines GS and Mesh—where GS provides realistic visual observations and an invisible Mesh provides physical collision detection—running in parallel within the Unity engine.

**Core Idea**: A complete video-to-interactive-simulation pipeline built on geometrically consistent GS reconstruction (scale-invariant depth/normal supervision + geometric consistency loss + screen-space covariance culling), a hybrid GS-Mesh representation, static/dynamic obstacle composition, and multi-level scene augmentation.

## Method

### Overall Architecture
Given a monocular video, Vid2Sim operates in two stages: (1) Geometrically consistent scene reconstruction—utilizing monocular depth/normal priors to regularize GS training, and extracting high-quality Mesh from GS via TSDF; (2) Realistic and interactive simulation construction—running a hybrid GS+Mesh representation in Unity, adding static obstacles and dynamic pedestrians, and performing diverse augmentations through scene editing and weather simulation.

### Key Designs

1. **Scale-Invariant Geometry Supervision**:

    - **Function**: Utilizes priors from monocular depth estimation models (e.g., Depth Anything v2) to improve the geometric reconstruction quality of GS.
    - **Mechanism**: Instead of using an L1 depth loss (since the scales of SfM-initialized GS and monocular depth predictions do not align), a patch-level Normalized Cross-Correlation (NCC) loss is employed: $\mathcal{L}_{depth} = 1 - \frac{1}{\|\mathcal{P}\|}\sum_{p}\sum_k \frac{\hat{D}'_{p,k} D'_{p,k}}{\hat{\sigma}_p \sigma_p}$, which evaluates local structural similarity rather than absolute scale. Normal supervision is applied using a cosine distance loss. An additional geometric consistency loss $\mathcal{L}_{geo}$ constrains adjacent pixel normals to be consistent (weighting regions with small depth gradients more heavily), while minimizing the shortest axis of the GS to approximate 2D disks.
    - **Design Motivation**: The NCC loss is insensitive to global scales and focuses solely on local structural alignment, avoiding training instability caused by scale mismatch. The geometric consistency loss further ensures surface smoothness.

2. **Screen-Space Covariance Culling**:

    - **Function**: Eliminates floater artifacts generated when the agent explores viewpoints that deviate significantly from the training views.
    - **Mechanism**: During rendering, the maximum norm of the covariance matrix $\|\Sigma'\|_\infty$ of each GS projected onto 2D is checked. If it exceeds an $\alpha$ proportion of the image area, the GS is culled. The formulation is $\|\Sigma'\|_\infty > \alpha \cdot A_{img}$. This is a simple, size-based filtering executed at runtime.
    - **Design Motivation**: In RL training, random exploration by the agent can lead to extreme angles never covered by training views (e.g., near the ground). In such cases, large Gaussians project as artifacts covering the entire screen, which are incorrectly perceived as obstacles and hinder navigation.

3. **Hybrid Representation & Interactive Composition**:

    - **Function**: Creates navigation training environments that are both photorealistic and physically interactive.
    - **Mechanism**: GS renders photorealistic RGB and depth observations in real-time using a custom Unity shader; the TSDF Mesh extracted from GS is set to invisible but handles collision detection. Static obstacles (traffic cones, trash cans, etc.) are placed randomly, with occlusion between foreground objects and the GS background handled via z-buffering. Dynamic pedestrians move through the scene using A* path planning. Scene augmentations include video-consistent style editing (lighting/seasonal changes) and particle system weather simulation (rain/fog/snow).
    - **Design Motivation**: The hybrid design, leveraging the strengths of both GS and Mesh, achieves an optimal balance between visual quality and physical interaction. Diverse obstacles and scene augmentations ensure the agent learns robust navigation policies.

### Loss & Training
Geometrically consistent reconstruction: $\mathcal{L}_{total} = \mathcal{L}_{rgb} + \mathcal{L}_{depth} + \mathcal{L}_{normal} + \mathcal{L}_{geo} + \mathcal{L}_{scale}$. A diverse urban scene dataset is reconstructed from 30 web videos. RL training employs the PPO algorithm, taking the robot's forward RGB camera observation and target direction/distance as the policy input.

## Key Experimental Results

### Main Results
Reconstruction quality and simulation capability comparison:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Real-time | Interactive | RL Training |
|------|-------|-------|--------|------|--------|--------|
| Instant-NGP | 27.50 | 0.827 | 0.240 | ✗ | ✗ | ✗ |
| 3DGS | 31.85 | 0.921 | 0.136 | ✓ | ✗ | ✗ |
| 2DGS | 30.82 | 0.915 | 0.154 | ✓ | ✗ | ✗ |
| Video2Game | 28.32 | 0.834 | 0.275 | ~✓ | ✓ | ✗ |
| **Vid2Sim** | **32.41** | **0.927** | **0.127** | **✓** | **✓** | **✓** |

Navigation task comparison:

| Method | Observation | PointNav SR↑ | SocialNav SR↑ |
|------|------|             |--------------|
| Mesh Simulation | RGB | 48.8% | 43.2% |
| Vid2Sim (Oracle) | Depth | 92.0% | 85.6% |
| **Vid2Sim (Full)** | **RGB** | **80.8%** | **Significant Improvement** |

### Ablation Study

| Configuration | PSNR↑ | Navigation SR↑ | Description |
|------|-------|---------|------|
| Base 3DGS | 31.85 | Low | No geometric regularization |
| + Depth/Normal Supervision | Improved | Improved | Improves geometry |
| + Geometric Consistency Loss | Improved | Improved | Smoother surface |
| + Covariance Culling | - | Further Improved | Eliminates exploration artifacts |
| + Obstacle Composition | - | **Highest** | More robust policy |

### Key Findings
- Agents trained in Vid2Sim achieve a 31.2% higher success rate in digital twins compared to those trained in Mesh simulation, and a 68.3% improvement in real-world deployment—demonstrating that realistic visual observations significantly reduce the sim2real gap.
- Geometrically consistent reconstruction outperforms baselines across all metrics; in particular, the improved normal rendering quality is crucial for collision detection.
- Screen-space covariance culling effectively eliminates exploration artifacts, solving a critical issue in RL training with a simple yet effective approach.
- Scene augmentations (style/weather) further enhance the generalization capability of the agent.

## Highlights & Insights
- **Complete Real2Sim Pipeline**: An end-to-end workflow from monocular videos to interactive simulations, resolving the entire pipeline from reconstruction to training and deployment.
- **Engineering Wisdom of Hybrid Representation**: GS is responsible for "seeing" while Mesh handles "collision," featuring a clear division of labor efficiently integrated in Unity.
- **Covariance Culling**: A simple size-threshold filter successfully resolves severe GS artifacts at extreme angles, showing that targeted solutions can be more effective than complex methodologies.

## Limitations & Future Work
- Reliance on SfM for camera pose estimation, which might fail in dynamic scenes and low-texture areas.
- Limited to urban ground navigation scenes, without generalization to indoor or aerial navigation.
- The quality of the Mesh extracted from GS still has room for improvement, especially for fine structures.
- Temporal consistency in scene editing may not be fully perfect.

## Related Work & Insights
- **vs Video2Game**: While both transition from video to interactive scenes, Video2Game is based on textured Mesh (PSNR 28.32), whereas Vid2Sim is based on GS (PSNR 32.41), leading to significantly superior visual quality.
- **vs 3DGS/2DGS**: Standard GS methods only render and cannot be interacted with. Vid2Sim's hybrid representation resolves this fundamental limitation.
- **vs Sim-on-Wheels**: Requires a continuous operation of real vehicles, which is costly. Vid2Sim constructs simulations using only videos.

## Rating
- Novelty: ⭐⭐⭐⭐ The hybrid scene representation and covariance culling are practical innovations, and the complete pipeline holds high engineering value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The three-level evaluation spanning reconstruction quality, simulation navigation, and real-world deployment is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear system organization with abundant and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Provides an extensible, high-quality solution for sim2real, and the 30-scene dataset holds significant value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation](roomtour3d_geometry-aware_video-instruction_tuning_for_embodied_navigation.md)
- [\[ICCV 2025\] RoboPearls: Editable Video Simulation for Robot Manipulation](../../ICCV2025/3d_vision/robopearls_editable_video_simulation_for_robot_manipulation.md)
- [\[CVPR 2025\] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](isegman_interactive_segment-and-manipulate_3d_gaussians.md)
- [\[CVPR 2025\] Towards Realistic Example-Based Modeling via 3D Gaussian Stitching](towards_realistic_example-based_modeling_via_3d_gaussian_stitching.md)
- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)

</div>

<!-- RELATED:END -->
