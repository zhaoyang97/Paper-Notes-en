---
title: >-
  [Paper Note] Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation
description: >-
  [CVPR2026][3D Vision][3D Gaussian Splatting] This paper proposes a Physics-Guided Score Distillation framework that leverages physics simulation (MPM) as a motion prior to guide Video-SDS optimization…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Dynamic Scene Editing"
  - "Weather Effects"
  - "Physics Simulation"
  - "Score Distillation Sampling"
  - "MPM"
date: 2026-05-08
content_hash: 4d9604b441761c21
---

# Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation

**Conference**: CVPR2026
**arXiv**: [2504.05296](https://arxiv.org/abs/2504.05296)  
**Code**: [Project Page](https://galfiebelman.github.io/let-it-snow/)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Dynamic Scene Editing, Weather Effects, Physics Simulation, Score Distillation Sampling, MPM

## TL;DR

This paper proposes a Physics-Guided Score Distillation framework that leverages physics simulation (MPM) as a motion prior to guide Video-SDS optimization, enabling the generation of dynamic weather effects (snow, rain, fog, sandstorm) with physically plausible motion and photorealistic appearance in static 3DGS scenes.

## Background & Motivation

**High demand for dynamic editing of static scenes**: 3D Gaussian Splatting enables efficient reconstruction of static scenes, but adding temporal dynamic effects such as weather remains a labor-intensive process with a high barrier to entry.

**Static editing methods cannot model temporal evolution**: Methods such as ClimateNeRF and GaussCtrl are limited to static appearance modification and cannot represent continuous particle emission and accumulation processes.

**Physics simulation lacks photorealistic appearance**: Physics-based methods such as PhysGaussian and PAC-NeRF provide physically plausible motion but cannot synthesize realistic appearances for newly introduced dynamic elements.

**Data-driven 4D generation yields uncontrollable motion**: Methods such as DreamGaussian4D and Animate124 rely on diffusion models for motion generation, resulting in incoherent motion in complex multi-particle scenes that require continuous particle emission.

**Fundamental tension between motion and appearance**: Physics simulation provides strong motion priors but lacks realism, while Video-SDS can generate photorealistic appearances but cannot independently learn complex motion — the two must be unified.

**Existing 4D editing methods operate on fixed Gaussian sets**: They do not support the continuous particle emission, accumulation, and removal mechanisms required for weather effects.

## Method

### Overall Architecture

The framework consists of two stages: (1) a Material Point Method (MPM) physics simulation generates reference motion trajectories as a prior; (2) a recurrent neural dynamics model is trained and jointly optimized for motion and appearance via Physics-Guided Score Distillation.

### Physics-Based Motion Prior

- After reconstructing the static scene with 3DGS, a mesh is extracted and static Gaussians are mapped to MPM particles as obstacles.
- Dynamic particles are introduced (with emission regions, rates, initial velocities, and material properties), and their motion trajectories are computed via MPM simulation.
- **Active particle tracking**: Particles that become stationary or leave the simulation boundary are automatically removed, enabling large-scale particle simulation.
- **Mesh collision refinement**: To compensate for MPM's coarse grid resolution, weather-specific collision handling is designed — snow is projected onto surfaces with interpolation from nearby Gaussians to achieve natural accumulation; rain uses a 3D humidity grid to track moisture with Gaussian smoothing and temporal decay; sand is displaced along surface normals with anisotropic scaling.

### Recurrent Neural Dynamics Model

- **Input**: Previous-timestep rendering state (position, rotation, appearance) + physics simulation velocity + time step.
- Separate MLPs are used for *active* and *collided* physical states.
- Positions and velocities are encoded with Fourier features; time is encoded with sinusoidal encoding.
- **Output**: velocity correction $\Delta \mathbf{v}$, angular velocity $\boldsymbol{\omega}$, and appearance increment $\Delta \mathcal{A}$.
- Motion update: $\mathbf{v}_g(t) = \mathbf{v}_g^{\text{init}}(t) + \Delta \mathbf{v}_g$, $\mathbf{x}_g(t) = \mathbf{x}_g(t{-}1) + \mathbf{v}_g(t)$.
- Appearance parameters are initialized by an LLM based on the weather text description.

### Loss & Training

The total loss is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Video-SDS}} + \lambda_{\text{xyz}}\mathcal{L}_{\text{xyz}} + \lambda_{\text{vel}}\mathcal{L}_{\text{vel}} + \lambda_{\text{rot}}\mathcal{L}_{\text{rot}} + \lambda_{\text{app}}\mathcal{L}_{\text{app}}$$

- **Video-SDS loss**: Provides photorealism supervision via a text-to-video diffusion model.
- **Position regularization** $\mathcal{L}_{\text{xyz}}$: L2 distance between the learned trajectory and the simulated trajectory.
- **Velocity regularization** $\mathcal{L}_{\text{vel}}$: L2 distance between the learned velocity and the simulated velocity.
- **Rotation regularization** $\mathcal{L}_{\text{rot}}$: Quaternion angular distance to prevent rotation drift.
- **Appearance regularization** $\mathcal{L}_{\text{app}}$: Penalizes large appearance increments to suppress recurrent accumulation errors.
- **Adaptive SDS weighting**: All regularization weights are dynamically scaled by $|\mathcal{L}_{\text{Video-SDS}}|$ — when the diffusion model is uncertain, physical guidance is strengthened; when it is confident, constraints are relaxed.

## Key Experimental Results

### Experimental Setup

- **Datasets**: MipNeRF 360 (Garden, Bicycle, Stump) + Tanks and Temples (Playground, Truck), totaling 5 scenes.
- **Weather effects**: Snow, rain, fog, sandstorm, plus creative text variants (purple snow, glittering yellow sand, magic particles).
- **Evaluation metrics**: CLIP_Sim, CLIP_Dir, VQAScore, ViCLIP-T, VE-Bench.

### Main Results (Table 1)

| Method | CLIP_Sim↑ | CLIP_Dir↑ | VQAScore↑ | ViCLIP-T↑ | VE-Bench↑ |
|--------|-----------|-----------|-----------|-----------|-----------|
| ClimateNeRF (F+S) | 0.23 | 0.07 | 0.87 | 0.15 | 0.28 |
| GaussCtrl (F+S) | 0.25 | 0.08 | 0.71 | 0.16 | 0.24 |
| **Ours (F+S)** | **0.29** | **0.12** | **0.92** | **0.20** | **0.45** |
| GaussCtrl (All) | 0.24 | 0.07 | 0.64 | 0.15 | 0.21 |
| **Ours (All)** | **0.28** | **0.11** | **0.89** | **0.19** | **0.41** |

The proposed method outperforms static editing baselines across all image and video metrics, with particularly notable improvement on VE-Bench (+61%).

### Ablation Study (Table 2)

| Variant | CLIP_Sim | CLIP_Dir | VQAScore | ViCLIP-T | VE-Bench |
|---------|----------|----------|----------|----------|----------|
| w/o collision handling | 0.24 | 0.10 | 0.83 | 0.16 | 0.34 |
| w/o appearance optimization | 0.25 | 0.10 | 0.82 | 0.16 | 0.37 |
| w/o motion simulation | 0.18 | 0.03 | 0.35 | 0.08 | 0.13 |
| w/o physics guidance | 0.26 | 0.10 | 0.85 | 0.17 | 0.37 |
| **Full method** | **0.28** | **0.11** | **0.89** | **0.19** | **0.41** |

### Key Findings

- **Removing motion simulation (w/o Motion) causes the most severe degradation**: Video-SDS alone cannot learn physically plausible motion for continuous multi-particle emission; VQAScore drops from 0.89 to 0.35.
- **Physics guidance is critical for joint optimization**: Fixing the physics motion without joint optimization (w/o PG) also impedes Video-SDS appearance optimization.
- **Collision handling is indispensable**: Without it, particles hover above surfaces and natural accumulation cannot be achieved.
- **Adaptive SDS weighting outperforms fixed weighting**: Fixed weights cause problems regardless of magnitude — too small yields noisy artifacts, too large over-constrains and prevents appearance refinement.
- Compared to the 4D editing baseline Instruct-4DGS, the latter lacks physical priors and produces incoherent motion (VQAScore 0.57 vs. 0.89).

## Highlights & Insights

- **Elegant core insight**: Physics simulation serves as a "soft-constrained motion prior" rather than a hard constraint, elegantly reconciling the tension between physical plausibility and visual realism.
- **Adaptive SDS weighting**: Dynamically modulates the strength of physical constraints based on diffusion model uncertainty, eliminating the need for laborious weight tuning.
- **General weather framework**: A single framework supports four distinctly different weather effects (snow, rain, fog, sandstorm) and responds to creative text prompts.
- **Comprehensive ablation**: Individual ablations of collision handling, motion/appearance optimization, physics guidance, adaptive weighting, individual regularization terms, and trajectory drift provide thorough empirical justification.

## Limitations & Future Work

- **No bidirectional interaction**: Dynamic particles do not induce deformation or motion in static scene elements.
- **Static Gaussian appearance is not updated**: Environmental lighting and shadow changes caused by weather are not reflected.
- **Trajectory drift threshold**: When the optimized trajectory deviates too far from the physics prior, the guidance velocity signal becomes unreliable.
- **Dependence on MPM simulation quality**: The quality of the physics prior directly affects the final results.
- **Computational cost**: The two-stage pipeline combining MPM simulation and Video-SDS optimization is computationally heavy.

## Related Work & Insights

- **Static 3D editing**: ClimateNeRF (ICCV'23) adds static weather effects without temporal dynamics; GaussCtrl (ECCV'24) enables text-guided Gaussian editing but is limited to appearance modification.
- **Physics-driven animation**: PhysGaussian (CVPR'24) uses MPM to animate existing scene elements but cannot synthesize appearances for new elements; RainyGS focuses on rain but does not generalize to other weather types.
- **4D generation/editing**: Methods such as Gaussians2Life and Instruct-4DGS perform 4D editing via Score Distillation but exhibit uncontrollable motion in multi-particle continuous emission scenarios.
- **Paper positioning**: This work is the first to unify physics simulation priors and Video-SDS within a single optimization framework, filling the gap in dynamic weather scene editing.

## Rating

- Novelty: ⭐⭐⭐⭐ — The unified framework of physics-prior-guided SDS is conceptually novel, and the adaptive SDS weighting design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 scenes × 4 weather types × multiple viewpoints, with highly detailed ablations covering collision handling, motion, appearance, physics guidance, adaptive weighting, individual regularization terms, and trajectory drift.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-integrated figures and tables, and well-articulated problem motivation.
- Value: ⭐⭐⭐⭐ — Opens a new direction for dynamic 3DGS editing; the framework is extensible to a broader range of dynamic effect scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing](rap_fast_feedforward_rendering-free_attribute-guided_primitive_importance_score_.md)
- [\[ICCV 2025\] Stable Score Distillation](../../ICCV2025/3d_vision/stable_score_distillation.md)
- [\[CVPR 2026\] Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields](dynamic_black-hole_emission_tomography_with_physics-informed_neural_fields.md)
- [\[CVPR 2026\] NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather](nimbusgs_unified_3d_scene_reconstruction_under_hybrid_weather.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)

</div>

<!-- RELATED:END -->
