---
title: >-
  [Paper Note] Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders
description: >-
  [NeurIPS 2025][3D Vision][Differentiable Physics Simulation] This paper proposes AS-DiffMPM, a differentiable Material Point Method (MPM) framework supporting arbitrary-shape rigid body colliders…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Differentiable Physics Simulation"
  - "System Identification"
  - "Material Point Method"
  - "3D Gaussian Splatting"
  - "Collision Handling"
date: 2026-05-08
content_hash: 4d8f0d5f26dc55eb
---

# Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders

**Conference**: NeurIPS 2025
**arXiv**: [2511.06846](https://arxiv.org/abs/2511.06846)
**Code**: [Available](https://as-diffmpm.github.io)
**Area**: 3D Vision
**Keywords**: Differentiable Physics Simulation, System Identification, Material Point Method, 3D Gaussian Splatting, Collision Handling

## TL;DR

This paper proposes AS-DiffMPM, a differentiable Material Point Method (MPM) framework supporting arbitrary-shape rigid body colliders, combined with multiple novel-view synthesis methods to enable system identification of physical parameters from visual observations.

## Background & Motivation

Estimating physical parameters of objects (e.g., viscosity, elastic modulus) from video observations has important applications in robotics, VR, and graphics. Existing methods (e.g., PAC-NeRF) combine differentiable rendering with differentiable MPM and optimize physical parameters from images via gradient descent. However, these methods have critical limitations:

**Collider Restriction**: Existing system identification methods support only planar colliders (e.g., flat ground), and cannot handle complex collisions between objects and non-planar surfaces.

**Non-differentiable MPM Simulators**: MPM simulators capable of handling arbitrary-shape colliders (e.g., CPIC) are non-differentiable and thus incompatible with gradient-based optimization.

3. In real-world scenarios, objects frequently collide with rigid bodies of various shapes (e.g., falling on a box edge, hitting a complex object); supporting only planar interactions severely restricts applicability.

## Method

### Overall Architecture

AS-DiffMPM operates in three stages:

1. **Rigid Collider Reconstruction**: Reconstruct rigid bodies from multi-view images using 2DGS, or directly use a mesh.
2. **Physics Simulation**: Simulate continuum material motion using AS-DiffMPM (differentiable MPM with arbitrary-shape collision handling).
3. **Rendering & Optimization**: Map simulated particle positions back to rendering primitives, and backpropagate gradients through photometric loss against real video to optimize physical parameters.

Key gradient propagation chain:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_p \left(\frac{\partial \mathcal{L}}{\partial I} \cdot \frac{\partial I}{\partial \mathbf{x}_p}\right) \cdot \frac{\partial \mathbf{x}_p}{\partial \theta}$$

where $\theta$ denotes physical parameters, $\mathbf{x}_p$ denotes particle positions, and $I$ denotes the rendered image.

### Key Designs

#### 1. MPM Fundamentals and Collision Handling

MPM three-stage cycle:
- **P2G (Particle-to-Grid)**: Particle mass and momentum are transferred to neighboring grid nodes via weight functions.
- **G-OP (Grid Operations)**: External forces (e.g., gravity) are applied on the grid and boundary conditions are enforced.
- **G2P (Grid-to-Particle)**: The grid velocity field is interpolated back to particles to update particle states.

Traditional methods handle collisions in the G-OP stage (zeroing out velocities of grid nodes inside colliders), but this fails for complex geometries — all particles within the same grid cell share a single velocity field.

#### 2. Differentiating the CPIC Method (Core Contribution)

AS-DiffMPM implements **particle-level collision handling** based on Compatible Particle-in-Cell (CPIC):

**Step 1 — Project Rigid Bodies onto Collision Grid**:
- Sample **rigid particles** $\mathbf{x}_{rp}$ on the rigid body surface.
- Identify **affinity grid nodes**: project nodes in the $3\times3\times3$ neighborhood of each rigid particle onto its associated face and check whether the projection falls within the face.
- Reconstruct the collision grid: each node stores an affinity flag $A_g$, unsigned distance $d_g$, side flag $T_g$, and normal $\mathbf{n}_g$.

**Step 2 — Collision Grid to Material Particles**:
- Interpolate collision attributes from grid to particles: $d_p = \sum_{g} w_g A_g T_g d_g$, $\mathbf{n}_p = \sum_g w_g A_g \mathbf{n}_g$.
- **Penetration Correction**: If a particle penetrates the boundary, apply a penalty force $\mathbf{f}_p = -k_h d_p \mathbf{n}_p$.
- **Compatibility Check**: Compare particle $T_p$ with neighboring node $T_g$; particles on opposite sides of the boundary are deemed incompatible.

**Step 3 — Collision Handling in P2G/G2P**:
- P2G: Particles transfer velocity only to **compatible nodes** (no collision handling needed in the G-OP stage).
- G2P: For incompatible nodes, directly reuse the current particle velocity and project it onto the surface: $\mathbf{v}_p^{proj} = \mathbf{v}_p - (\mathbf{v}_p \cdot \mathbf{n}_p)\mathbf{n}_p$.

#### 3. 2D Gaussians as Colliders

The paper innovatively supports **rigid bodies reconstructed via 2D Gaussian Splatting** as colliders. Each 2D Gaussian corresponds to a planar disk, serving as a collision primitive $\xi^i$ in place of mesh faces. The remainder of the pipeline is identical to mesh-based colliders; the only difference is that affinity node identification requires checking whether the projection falls within the disk.

#### 4. Rendering–Physics Integration

Three rendering methods are supported:
- **Voxel-based NeRF (DVGO)**: Requires an additional voxel-to-particle mapping.
- **2D Gaussian Splatting (2DGS)**: Naturally aligned with MPM (both are particle-based representations).
- **MDyn-3DGS**: Performs dynamic scene reconstruction first, then optimizes physical parameters.

### Loss & Training

- **Particle Trajectory Supervision**: Particle-level MSE loss.
- **Visual Observation Supervision**: Photometric loss between rendered images and real frames.
- Adam optimizer; physical parameters are iteratively optimized from an initial guess.
- Dataset: 10 sequences per collider/material combination (16 time steps), 11 camera viewpoints.

## Key Experimental Results

### Main Results (Particle Trajectory Supervision)

System identification from particle trajectories (absolute error ×100):

| Material Type | Collider | Parameter | RP-DiffMPM | GOP-DiffMPM | **AS-DiffMPM** |
|---------|--------|------|:----------:|:-----------:|:-------------:|
| Newtonian Fluid | Bunny | log₁₀(μ) | 6.69±5.97 | 4.35±5.24 | **0.43±0.30** |
| Newtonian Fluid | Armadillo | log₁₀(κ) | 34.61±46.5 | 0.22±0.24 | **0.22±0.24** |
| Non-Newtonian Fluid | Box | log₁₀(τ_Y) | 20.13±25.8 | 7.28±3.75 | **5.86±3.21** |
| Granular | Bunny | θ_fric | 1.91±0.42 | 0.06±0.07 | **0.11±0.15** |

### Visual Observation Supervision

System identification results with different rendering methods combined with AS-DiffMPM:

| Material | Collider | Parameter | DVGO | 2DGS | **MDyn-3DGS** |
|------|--------|------|:----:|:----:|:------------:|
| Newtonian Fluid | Armadillo | log₁₀(μ) | 1.99±1.59 | 5.28±2.55 | **0.97±0.86** |
| Newtonian Fluid | Armadillo | log₁₀(κ) | 32.94±42.2 | 49.66±35.8 | **14.99±17.1** |
| Granular | Bunny | θ_fric | 4.02±0.68 | **0.50±0.29** | 3.33±0.62 |
| Granular | Armadillo | θ_fric | 4.11±1.05 | **0.36±0.13** | 2.89±0.78 |

### Ablation Study

Comparison of three collision handling strategies:
- **RP-DiffMPM** (rigid particles): Consistently worst-performing; lacks a dedicated collision strategy.
- **GOP-DiffMPM** (grid-level SDF): Limited accuracy on complex geometries (shared velocity field per grid cell).
- **AS-DiffMPM** (particle-level CPIC): Optimal or on-par in the majority of scenarios.

### Key Findings

1. Particle-level collision handling (CPIC) significantly outperforms grid-level methods in system identification.
2. MDyn-3DGS generally performs best (leveraging dynamic reconstruction as an additional constraint), but 2DGS is superior for granular materials.
3. Non-Newtonian fluids are the most challenging; no single rendering method consistently leads across all colliders.
4. System identification with complex colliders is inherently more difficult than with planar ones, but is critical for advancing toward models with fewer assumptions.

## Highlights & Insights

- **Fills an Important Gap**: The first work to achieve differentiable MPM system identification with arbitrary-shape colliders.
- **Unified Collision Interface**: Supports both mesh and 2D Gaussian collider representations as plug-and-play modules.
- **Rendering-Agnostic**: Compatible with NeRF, 3DGS, 2DGS, and other rendering methods.
- **Real-World Validation**: Successfully estimates non-Newtonian material parameters in a real experiment of dough colliding with a box edge.

## Limitations & Future Work

1. Gaussians drift from their original positions after long simulations, causing visual artifacts — time-varying Gaussian attributes could be introduced to address this.
2. Colliders are currently assumed to be static rigid bodies; deformable or dynamic colliders are not supported.
3. Parameter estimation for non-Newtonian fluids still exhibits high variance; stronger regularization is needed.
4. The benchmark of 75 synthetic sequences is relatively small; validation on larger-scale real-world data is required.

## Related Work & Insights

- **PAC-NeRF**: Pioneered NeRF + DiffMPM for system identification, but limited to planar collisions.
- **PhysGaussian**: Drives 3DGS physical animation with MPM, but does not perform system identification.
- **CPIC (hu2018moving)**: The original non-differentiable arbitrary-collider MPM method; the foundation for this paper's differentiable extension.
- Insight: End-to-end pipelines combining differentiable physics engines with differentiable rendering represent a key pathway toward understanding the physical world.

## Rating

- Novelty: ⭐⭐⭐⭐ Differentiable MPM with arbitrary colliders is a substantive technical advance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three material types × three colliders × three rendering methods; broad coverage.
- Writing Quality: ⭐⭐⭐ Technical details are thorough but notation-dense; requires considerable background knowledge.
- Value: ⭐⭐⭐⭐ Substantially extends the applicability of differentiable physics simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[ICCV 2025\] TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update](../../ICCV2025/3d_vision/2d_gaussian_splattingbased_sparseview_transparent_object_dep.md)
- [\[CVPR 2026\] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](../../CVPR2026/3d_vision/motionanymesh_physicsgrounded_articulation_for_sim.md)
- [\[NeurIPS 2025\] EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes](eag3r_event-augmented_3d_geometry_estimation_for_dynamic_and_extreme-lighting_sc.md)
- [\[ICLR 2026\] Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting](../../ICLR2026/3d_vision/augmented_radiance_field_a_general_framework_for_enhanced_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
