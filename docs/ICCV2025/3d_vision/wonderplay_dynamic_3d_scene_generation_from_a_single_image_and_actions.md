---
title: >-
  [Paper Note] WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions
description: >-
  [ICCV 2025][3D Vision][dynamic 3D scene generation] WonderPlay introduces a Hybrid Generative Simulator that combines coarse 3D dynamic simulation from a physics solver with high-quality generation from a video diffusion…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "dynamic 3D scene generation"
  - "physics simulation"
  - "video generation"
  - "hybrid generative simulator"
  - "single-image interaction"
date: 2026-05-08
content_hash: 710e57934e459c0b
---

# WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions

**Conference**: ICCV 2025
**arXiv**: [2505.18151](https://arxiv.org/abs/2505.18151)  
**Code**: [https://kyleleey.github.io/WonderPlay/](https://kyleleey.github.io/WonderPlay/) (coming soon)  
**Area**: 3D Vision
**Keywords**: dynamic 3D scene generation, physics simulation, video generation, hybrid generative simulator, single-image interaction

## TL;DR

WonderPlay introduces a Hybrid Generative Simulator that combines coarse 3D dynamic simulation from a physics solver with high-quality generation from a video diffusion model, enabling realistic multi-material dynamic 3D scene generation from a single image and user-specified actions. The framework supports diverse material types including rigid bodies, cloth, liquids, smoke, and granular materials.

## Background & Motivation

**Background**: Dynamic 3D scene generation is a core requirement for AR/VR and embodied AI. Existing approaches fall into two main categories: physics-based simulation methods and conditional video generation methods.

**Limitations of Prior Work**:
- Physics simulation methods (e.g., PhysGaussian, PhysDreamer) require accurate physics solvers and complete 3D physical state reconstruction. However, reconstructing the full physical state of snow, sand, cloth, or fluid from a single image is practically infeasible, limiting these methods to rigid bodies and simple elastic objects.
- Video generation methods (e.g., CogVideoX, Sora) can produce visually realistic videos of physical phenomena but cannot accept precise 3D actions as input and thus lack controllability.

**Key Challenge**: Physics simulation offers accurate action response but suffers from poor visual quality and limited material coverage; video generation achieves high visual fidelity but lacks action controllability.

**Goal**: Starting from a single image and accepting 3D physical action inputs (gravity, wind fields, point forces), how can one generate realistic dynamic 3D scenes spanning diverse material types?

**Key Insight**: Redefine the roles of the physics simulator and the video generator — the physics simulator provides coarse but controllable motion guidance, while the video generator is responsible for refining motion and visual quality.

**Core Idea**: Use the physics simulator to produce coarse 3D dynamics as conditioning signals to drive a video diffusion model toward realistic video generation, then inversely update the 3D scene from the generated video, forming a closed loop.

## Method

### Overall Architecture

**Input**: A single image $\mathbf{I}$ and 3D actions (gravity $\mathbf{f}_g$, wind field $\mathbf{f}_w(x,y,z,t)$, point force $\mathbf{f}_p(t)$). **Output**: A dynamic 3D scene sequence $\{\mathcal{S}_t\}_{t=0}^T$.

The pipeline consists of three stages:
1. **3D Scene Reconstruction**: Reconstruct the initial 3D scene $\mathcal{S}_0$ from the single image.
2. **Hybrid Generative Simulation**: Physics solver generates coarse dynamics → conditional video generation → video inversely updates the 3D scene.
3. **Output Dynamic 3D Scene**: Renderable from arbitrary viewpoints.

### Key Designs

1. **3D Scene Representation and Reconstruction**

    - **Function**: Reconstruct a 3D scene containing background and foreground objects from a single image.
    - **Mechanism**: The background is represented using FLAGS (Fast Layered Gaussian Surfels); foreground objects are represented as "topological Gaussian Surfels," which augment standard Gaussian Surfels with a connectivity edge matrix $\mathbf{E} \in \{0,1\}^{N_O \times N_O}$ and velocity $\mathbf{v}_t$. Object meshes are generated from segmented images via InstantMesh and Gaussian Surfels are bound to each mesh vertex.
    - **Design Motivation**: Topological connectivity allows objects to be directly simulated by physics solvers; separating background and foreground objects enables different control strategies to be applied to each.

2. **Hybrid Generative Simulator**

    - **Function**: Core innovation module that fuses physics simulation and video generation to predict dynamics.
    - **Mechanism**:
        - First, a physics solver (Genesis framework, supporting coupled multi-material solvers) computes coarse dynamic scenes $\{\tilde{\mathcal{S}}_t\}$: $\mathbf{v}_{t+1}, \mathbf{p}_{t+1}^O, \mathbf{q}_{t+1}^O = \text{solver}(\tilde{\mathcal{S}}_t, \mathbf{f}_g, \mathbf{f}_w(t), \mathbf{f}_p(t))$
        - The coarse dynamic motion and appearance signals are then fed into a video generator: $\mathbf{V} = g(\mathbf{F}, \tilde{\mathbf{V}}, \mathbf{I})$
        - Finally, the generated video updates the coarse 3D scene via differentiable rendering.
    - **Design Motivation**: The physics solver's reconstruction and simulation are imprecise, but they provide the correct directional response to actions; the video generator, trained on large-scale video data, encodes rich physical priors that supply fine-grained motion and appearance details.

3. **Bimodal Control**

    - **Function**: Jointly control the video generator using both motion signals and appearance signals.
    - **Mechanism**:
        - **Motion control**: Adopts the noise-warping strategy from Go-with-the-Flow, converting physics-simulated optical flow $\mathbf{F}$ into structured noise $\mathbf{N}(\mathbf{F})$ via iterative warping: $\mathbf{N}_{t+1} = \text{warp}(\mathbf{N}_t, \mathbf{F}_{t+1})$
        - **Appearance control**: Applies SDEdit, starting denoising from diffusion step $s_1 < S$: $\mathbf{V}_{s_1} = \alpha_{s_1}\tilde{\mathbf{V}} + \sqrt{1-\alpha_{s_1}^2}\mathbf{N}(\mathbf{F})$
    - **Design Motivation**: Using only motion signals leads to hallucinations (e.g., background texture changes); using only appearance signals loses fine-grained dynamics. Combining both preserves motion consistency and appearance consistency simultaneously.

4. **Spatially Varying Responsibility**

    - **Function**: Assign different generation responsibilities to the background and foreground dynamic objects.
    - **Mechanism**: Introduces a binary mask $\mathbf{M}$, blending at an additional denoising step $s_2 < s_1$: $\hat{\mathbf{V}}_{s_2} = \mathbf{M} \odot \mathbf{V}_{s_2} + (1-\mathbf{M}) \odot (\alpha_{s_2}\tilde{\mathbf{V}} + \sqrt{1-\alpha_{s_2}^2}\mathbf{N}(\mathbf{F}))$
    - **Design Motivation**: The background is generally static and should trust the physics simulation output more than the video generator, so as to prevent the generator from hallucinating nonexistent objects in background regions.

### Loss & Training

During the scene dynamic update stage, a photometric L1 loss is used: $\min_{\{\mathbf{c}_t^B, \mathcal{O}_t\}} \|\mathbf{V} - \tilde{\mathbf{V}}\|_1$, optimizing the motion trajectory and appearance of foreground objects while updating background colors to capture lighting effects.

## Key Experimental Results

### Main Results

Comparisons against physics simulation and conditional video generation methods are conducted on 15 scenes.

| Method | Imaging↑ | Aesthetic↑ | Motion↑ | Consistency↑ | PhysReal↑ |
|------|----------|------------|---------|--------------|-----------|
| PhysGen | 0.692 | 0.593 | 0.992 | 0.212 | 0.545 |
| PhysGaussian | 0.492 | 0.564 | 0.994 | 0.206 | 0.350 |
| CogVideoX | 0.686 | 0.574 | 0.993 | 0.219 | 0.670 |
| Tora | 0.644 | 0.620 | 0.992 | 0.210 | 0.530 |
| **WonderPlay (Ours)** | **0.695** | 0.610 | **0.995** | 0.217 | **0.700** |

### User Study (2AFC, 200 Participants)

| Comparison | Physical Plausibility↑ | Motion Fidelity↑ | Visual Quality↑ |
|----------|----------------|----------------|--------------|
| vs PhysGen | 78.0% | 78.0% | 80.1% |
| vs PhysGaussian | 80.2% | 81.2% | 85.2% |
| vs Tora | 77.0% | 72.0% | 71.0% |
| vs CogVideoX | 80.2% | 73.0% | 74.6% |

### Key Findings
- Video generation methods, despite high visual quality, struggle to follow physical action instructions (CogVideoX even fails to generate plausible dynamics for a duck falling into water).
- Physics simulation methods are limited to rigid/elastic bodies and cannot handle complex effects such as water surface reflections.
- WonderPlay achieves over 70–80% user preference across all three evaluation dimensions.
- Ablation studies confirm that removing motion signals causes loss of detailed dynamics; removing appearance signals induces hallucinations; spatially varying control effectively reduces background hallucinations.

## Highlights & Insights

- **Closed-Loop Design**: The closed loop of physics simulation → video generation → 3D update is highly elegant. The physics simulation does not need to be precise; it only needs to provide the correct directional guidance, while the video generation model fills in the details. This "coarse-to-fine" paradigm is transferable to many scenarios requiring integration of physical priors with data-driven methods.
- **Bimodal Control + Spatial Variation**: Decomposing control signals into motion and appearance modalities and assigning different levels of "trust" based on spatial region is a highly practical design pattern.
- **Material Generality**: A single framework covering rigid bodies, cloth, liquids, gases, and granular materials is rarely achieved in prior work.

## Limitations & Future Work

- Users must manually specify the material type of objects (6 categories); automatic inference is not supported.
- Physics simulation accuracy is bounded by the quality of initial 3D reconstruction, which is inevitably imprecise from a single image.
- Only three action types are supported (gravity, wind force, point force), making it difficult to express more complex interactions.
- Generation speed may be slow (960-step physics simulation + video diffusion).

## Related Work & Insights

- **vs PhysMotion**: PhysMotion also combines a physics solver with a video generator, but delegates all dynamics to the physics solver and uses the video generator solely for appearance refinement. WonderPlay assigns joint responsibility for dynamics to both components, thereby supporting a wider range of material types.
- **vs PhysGen**: PhysGen supports only 2D rigid body simulation, whereas WonderPlay supports 3D multi-material simulation.
- **vs CogVideoX/Tora**: These conditional video generation methods lack physical action controllability; WonderPlay provides precise action response through physics simulation.

## Rating

- Novelty: ⭐⭐⭐⭐ The hybrid generative simulator idea is creative, though individual components (physics simulation, video diffusion, SDEdit) are all existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative evaluation, user study, and ablation are fairly complete, but the sample size of 15 scenes is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, figures are well-designed, and the method is explained in a well-structured manner.
- Value: ⭐⭐⭐⭐ Provides an important reference for the direction of interactive 3D world models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] A Recipe for Generating 3D Worlds from a Single Image](a_recipe_for_generating_3d_worlds_from_a_single_image.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)

</div>

<!-- RELATED:END -->
