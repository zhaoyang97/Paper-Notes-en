---
title: >-
  [Paper Note] DRAWER: Digital Reconstruction and Articulation with Environment Realism
description: >-
  [CVPR 2025][Robotics][digital twin] The DRAWER framework automatically constructs interactive digital twins from static scene videos. By combining a dual scene representation of SDF and Gaussian Splatting, it achieves high-fidelity rendering and precise geometry. It supports articulation identification and simulation, Unreal Engine game creation, and real-to-sim-to-real robotic policy transfer.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "digital twin"
  - "articulated objects"
  - "Gaussian splatting"
  - "SDF"
  - "game engine"
  - "real-to-sim-to-real"
  - "robotic manipulation"
date: 2026-05-08
content_hash: 1adc78b100452b2f
---

# DRAWER: Digital Reconstruction and Articulation with Environment Realism

**Conference**: CVPR 2025  
**arXiv**: [2504.15278](https://arxiv.org/abs/2504.15278)  
**Institution**: UIUC / University of Washington / Allen Institute for AI / Cornell University  
**Area**: Robotics / 3D Reconstruction  
**Keywords**: digital twin, articulated objects, Gaussian splatting, SDF, game engine, real-to-sim-to-real, robotic manipulation  

## TL;DR
The DRAWER framework automatically constructs interactive digital twins from static scene videos. By combining a dual scene representation of SDF and Gaussian Splatting, it achieves high-fidelity rendering and precise geometry. It supports articulation identification and simulation, Unreal Engine game creation, and real-to-sim-to-real robotic policy transfer.

## Background & Motivation

**Background**: Creating virtual digital replicas from real-world data holds significant potential in gaming, robotics, and virtual reality. Existing methods either focus solely on appearance modeling while neglecting physical interaction, or prioritize interactivity at the expense of photorealism.

**Limitations of Prior Work**: 
   - **NeRF/3DGS methods**: Good rendering quality but insufficient geometric accuracy; "floater" Gaussians are not aligned with the underlying geometry.
   - **Neural SDF methods**: Good geometric accuracy but lagged rendering quality, and volume rendering is slow.
   - **URDFormer**: Can estimate articulation but relies on predefined asset libraries, limiting physical realism.
   - No method simultaneously achieves: high-fidelity rendering + precise geometry + physical interaction + real-time performance.

**Key Challenge**: The conflict between appearance fidelity and geometric accuracy, and the gap between static reconstruction and interactivity.

**Key Insight**: A dual scene representation—using SDF for geometric accuracy and Gaussian Splatting for rendering quality; decomposing the scene into interactive components and automatically inferring articulation types and joint locations.

**Core Idea**: SDF + Gaussian dual representation (geometry + appearance) + articulation inference + amodal shape completion = complete interactive digital twin.

## Method

### Overall Architecture
Input multi-view posed images (from a single video) $\rightarrow$ dual scene representation reconstruction $\rightarrow$ articulated body identification and inference $\rightarrow$ amodal shape estimation + hidden area texture generation $\rightarrow$ interactive digital twin.

### Key Designs

1. **Dual Scene Representation**

    - **Neural SDF Branch**:
        - Maps 3D points and viewing directions to RGB color and signed distance values.
        - Supervised via volume rendering.
        - Provides precise geometry (high-quality mesh extraction).
    - **Gaussian Splatting Branch**:
        - Real-time rendering (>30 FPS).
        - Rasterization-based rendering, avoiding point-by-point sampling.
        - Provides high-fidelity appearance.
    - **Coupling Strategy**: The positions and normals of Gaussians are constrained by the SDF surface to ensure geometric consistency.

2. **Articulation Inference Module**

    - **Articulation Type Identification**: Distinguishes between revolute and prismatic articulation types.
    - **Joint Position Estimation**: Infers the position and orientation of the joint axis.
    - Compared with 3DOI (a foundation model for articulation prediction), the EA-Score reaches 0.994 vs. 0.861.

3. **Amodal Shape Estimation and Hidden Texture Generation**

    - **Function**: Reconstructs the shape and texture of occluded parts of objects.
    - **Key Challenge**: The internal surfaces revealed after opening drawers/doors are invisible in the original video.
    - **Solution**: Uses SDF for amodal shape completion + texture inpainting.
    - **Effect**: Creates a complete interactive object model.

4. **Game Engine Integration**

    - Features automatic export to Unreal Engine.
    - Supports physical collision, shooting interaction, and open/close animations.
    - Runs in real-time.

### Application Examples

1. **Interactive Games**
    - Free movement in first-person perspective.
    - Shooting spheres to generate realistic collisions.
    - Opening/closing drawers and cabinet doors.

2. **Real-to-Sim-to-Real Robotic Transfer**
    - Reconstructed scenes are imported into Isaac Sim.
    - Generates training data via motion planning.
    - Learns policies with 3D Diffusion Policy.
    - Directly transfers to a physical Franka Emika Panda robot.

## Key Experimental Results

### Articulation Inference Comparison

| Method | Total Objects | Correct Predictions↑ | Revolute Objects | Correct Revolute↑ | EA-Score↑ |
|------|---------|----------|-----------|----------|----------|
| 3DOI | 80 | 78 | 59 | 57 | 0.861 |
| **DRAWER** | 80 | **78** | 59 | **58** | **0.994** |

### End-to-End Pipeline Comparison (vs. URDFormer / Digital Cousin)

| Method | Precision | Recall | Visual Fidelity | Geometric Accuracy |
|------|-----------|--------|-----------|---------|
| URDFormer | Medium (relies on asset library) | Low | Medium | Low |
| Digital Cousin | Medium | Medium | Medium | Medium |
| **DRAWER** | **High** | **High** | **High** | **High** |

### Articulation Motion Simulation
- Compared with KlingAI, using Earth Mover's Distance to measure the quality of motion trajectories.
- DRAWER's physical simulation motion trajectories are "an order of magnitude better" than those generated by KlingAI.

### Ablation Study

| Component | Rendering Quality | Geometric Quality | Interaction Compatibility |
|------|---------|---------|-----------|
| SDF mesh only | Medium | High | ✓ |
| + Gaussian splatting | High | High | ✓ |
| + Articulation inference | High | High | ✓ (movable) |
| + Amodal completion | High | High | ✓ (complete) |

## Highlights & Insights
- **Dual representation** (SDF + Gaussian) leverages mutual strengths: SDF's geometric accuracy plus Gaussian's rendering speed and quality.
- **End-to-end fully automatic**: Goes from video to interactive digital twin without human intervention.
- **Real-to-Sim-to-Real** closed-loop validation: Demonstrates the practical value of the method in robotics.
- **EA-Score of 0.994** (near-perfect) indicates extremely accurate articulation inference.
- Establishes a complete pipeline from academic prototypes (scene reconstruction) to practical applications (gaming/robotics).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hearing Anywhere in Any Environment](hearing_anywhere_in_any_environment.md)
- [\[ICLR 2026\] ArtVIP: Articulated Digital Assets of Visual Realism, Modular Interaction, and Physical Fidelity for Robot Learning](../../ICLR2026/robotics/artvip_articulated_digital_assets_of_visual_realism_modular_interaction_and_phys.md)
- [\[CVPR 2025\] ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping](zerograsp_zero-shot_shape_reconstruction_enabled_robotic_grasping.md)
- [\[CVPR 2025\] RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins](robotwin_dual-arm_robot_benchmark_with_generative_digital_twins.md)
- [\[ICLR 2026\] Capturing Visual Environment Structure Correlates with Control Performance](../../ICLR2026/robotics/capturing_visual_environment_structure_correlates_with_control_performance.md)

</div>

<!-- RELATED:END -->
