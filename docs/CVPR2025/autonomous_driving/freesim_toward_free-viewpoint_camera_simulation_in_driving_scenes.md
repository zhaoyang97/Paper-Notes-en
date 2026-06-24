---
title: >-
  [Paper Note] FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes
description: >-
  [CVPR 2025][Autonomous Driving][Free-viewpoint simulation] This paper proposes FreeSim, which reformulates the challenging off-trajectory novel view generation problem as a generative image enhancement task. Combined with training data construction via piece-wise Gaussian reconstruction and a progressive view expansion strategy, it achieves high-quality free-viewpoint rendering with more than 3 meters of lateral offset in driving scenes for the first time.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Free-viewpoint simulation"
  - "3D Gaussian Splatting"
  - "diffusion models"
  - "progressive reconstruction"
  - "image enhancement"
  - "driving simulation"
date: 2026-05-08
content_hash: 756245f03311e092
---

# FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes

**Conference**: CVPR 2025  
**arXiv**: [2412.03566](https://arxiv.org/abs/2412.03566)  
**Code**: [drive-sim.github.io/freesim](https://drive-sim.github.io/freesim)  
**Area**: Autonomous Driving  
**Keywords**: Free-viewpoint simulation, 3D Gaussian Splatting, diffusion models, progressive reconstruction, image enhancement, driving simulation

## TL;DR

This paper proposes FreeSim, which reformulates the challenging off-trajectory novel view generation problem as a generative image enhancement task. Combined with training data construction via piece-wise Gaussian reconstruction and a progressive view expansion strategy, it achieves high-quality free-viewpoint rendering with more than 3 meters of lateral offset in driving scenes for the first time.

## Background & Motivation

### Background
Realistic driving simulation is a fundamental component of autonomous driving development. With the advent of 3D Gaussian Splatting (3DGS), reconstruction-based driving scene simulation has made rapid progress (e.g., PVG, StreetGS), achieving high-quality rendering along recorded trajectories.

### Limitations of Prior Work
1. **Sharp decline in rendering quality for off-trajectory views**: Existing methods can only render high-quality images along the recorded trajectory. When the viewpoint deviates from the recorded trajectory (e.g., a 3-meter lateral offset to simulate lane changing), the rendering results suffer from severe degradation (ghosting, blurring, distortion).
2. **Lack of off-trajectory ground truth (GT) data**: Vehicles can only drive along a single trajectory, making it impossible to simultaneously capture images from multiple parallel trajectories. Consequently, the training samples required by traditional multi-view generation models (e.g., trained on datasets like RealEstate10k or CO3D) are unavailable.
3. **Limitations of existing generative methods**: Purely generative methods such as SGD only handle rotational variations, FreeVS is constrained by LiDAR coverage, and UniSim, supervised by GANs, shows limited effectiveness at large offsets.

### Key Challenge
How to generate high-quality, consistent free-viewpoint rendered images without off-trajectory GT data?

### Key Insight
If the viewpoint offset is small, the rendered results from the reconstructed radiance field can still be recognizable despite some degradation. Restoring a high-quality image from a mildly degraded one is much easier than directly generating a novel view conditioned on pose transformation. Therefore, novel view generation is reformulated as an **image enhancement** task.

### Core Idea
A two-step approach: (1) Construct paired "degraded-high quality" training data to train a generative enhancement model (utilizing piece-wise Gaussian reconstruction + extrapolation rendering + Gaussian perturbation). (2) Progressively add off-trajectory viewpoints into the reconstruction training set, expanding step-by-step from small offsets to large offsets to ensure the rendering of each newly added viewpoint has only minor degradation.

## Method

### Overall Architecture
FreeSim is a hybrid "generation-reconstruction" system. For the generation part: a ControlNet enhancement model is trained based on the constructed training data to restore degraded rendering into high-quality images. For the reconstruction part: based on PVG, the generated images of off-trajectory viewpoints are progressively added to the training set for reconstruction, gradually expanding from small offsets (0.5m) to large offsets (3m+).

### Key Designs

#### 1. Training Data Construction

- **Function**: Constructs matched "degraded image - high quality image" training pairs to train the enhancement model in the absence of off-trajectory GT.
- **Mechanism**:
    - **Piece-wise Gaussian reconstruction**: Segments each complete Waymo trajectory into multiple short sub-segments (20 frames). Each sub-segment quickly reconstructs a small-scale Gaussian field (< 2 minutes per segment, at most 1M primitives), enabling the reconstruction of the entire Waymo dataset (1150 scenes) in just 40 hours on 8 GPUs.
    - **Extrapolation rendering to simulate degradation**: Keeps the last 4 frames of each sub-segment as test frames for extrapolation rendering (instead of interpolation) to simulate the degradation patterns of off-trajectory viewpoints (lateral movement of side cameras along the direction of travel $\approx$ lateral offset of the front camera).
    - **Gaussian perturbation to enhance diversity**: Randomly samples a portion of Gaussian primitives and applies translation noise (up to 0.2m) and rotation noise (up to 15°) to simulate degradation patterns like "target ghosting".
    - In total, approximately **1.5 million** training samples are constructed.
- **Design Motivation**: Direct multi-view generation conditioned on pose transformation is infeasible due to the lack of multi-trajectory data. By reformulating the task as image enhancement, it only requires constructing training data that matches the degradation patterns of off-trajectory views. Piece-wise reconstruction is both efficient and naturally produces extrapolation degradation patterns.

#### 2. Generative Enhancement Model

- **Function**: Enhances degraded rendering images to high-quality images.
- **Mechanism**:
    - Based on Stable Diffusion v1.5, integrating two ControlNet branches: one handles the degraded image condition $\mathbf{I}_d$, and the other handles the optional sparse LiDAR projection condition $\mathbf{I}_l$.
    - The output features of the two ControlNets are summed and fused into each resolution layer of the UNet.
    - Cross-attention for CLIP text embeddings is removed.
    - Image blending strategy: During training, the degraded image is mixed with the GT with a probability of 0.1 ($\alpha=0.5$) to prevent the model from only learning to repair severe degradations.
- **Design Motivation**: The degraded image provides a strong prior (nearly correct geometric structure), and the LiDAR condition supplements accurate near-range depth information. The dual-ControlNet design allows both conditions to be encoded independently and combined flexibly.
- **Loss & Training**: Standard diffusion training loss $\mathcal{L} = \mathbb{E}[\|\epsilon_\theta(z_t; c_d, c_l, t) - \epsilon\|_2^2]$

#### 3. Progressive Reconstruction

- **Function**: Expands the enhancement capability from small offsets to large offsets, avoiding direct processing of severely degraded rendering.
- **Mechanism**:
    - First, perform pre-reconstruction on the recorded trajectory using standard PVG.
    - Every 5k iterations, shift all viewpoints laterally by one step (default 0.5m), generate novel viewpoint images using the enhancement model, and add them to the training set.
    - Freeze the new training set, optimize the Gaussian field until near convergence, and then shift again.
    - Repeat this process to progressively expand from the recorded trajectory to off-trajectory viewpoints.
    - Finally, apply the enhancement model as post-processing to the rendering results to eliminate rolling shutter distortion and minor blurring caused by generative randomness.
- **Design Motivation**: If rendering is performed directly at a large offset, the image might be completely ruined, making it impossible for the enhancement model to restore it effectively. The progressive strategy ensures that each newly added viewpoint has only minor degradation, which the enhancement model can handle with ease.

## Key Experimental Results

### Main Results — Waymo Open Dataset

| Method | Trajectory PSNR↑ | Offset 1m FID↓ | Offset 2m FID↓ | Offset 3m FID↓ |
|------|---------------|------------|------------|------------|
| StreetGS | 28.01 | 25.8 | 35.4 | 47.6 |
| EmerNeRF | 29.18 | 32.3 | 40.2 | 49.8 |
| PVG (baseline) | 29.19 | 22.9 | 34.3 | 47.5 |
| **Ours** | 28.32 | **14.6** | **17.0** | **18.6** |

FreeSim achieves an FID of only 18.6 at a 3m offset, reducing it by **60.8%** compared to the PVG baseline.

### Ablation Study

| Configuration | @1m FID | @2m FID | @3m FID |
|------|---------|---------|---------|
| Non-progressive | 20.1 | 26.3 | 29.7 |
| w/o LiDAR | 15.5 | 18.5 | 21.3 |
| Step size 1.0m | 14.5 | 16.9 | 18.4 |
| Default (0.5m) | **14.6**| **17.0**| **18.6**|

### Key Findings

1. **Progressive reconstruction is crucial**: Without the progressive strategy, the FID at 3m offset deteriorates from 18.6 to 29.7, indicating that directly processing large-offset degradation is infeasible.
2. **LiDAR condition is more important for large offsets**: At a 3m offset, the FID without LiDAR increases from 18.6 to 21.3, whereas the difference is smaller at 1m.
3. **Step size $\le$ 1m typically yields good results**: Too large a step size (e.g., 1.5m) worsens performance at large offsets.
4. **The enhancement model generalizes well across methods**: Although the training data is constructed using only PVG, it can be applied to degraded renderings from other reconstruction methods like StreetGS.
5. Post-enhancement processing can effectively eliminate blur in high-frequency regions (such as trees and nearby vehicles).

## Highlights & Insights

1. **Ingenious reformulation of the problem**: Reformulating "off-trajectory novel view generation" as "degraded image enhancement" bypasses the core difficulty of lacking multi-trajectory GT and simplifies the problem to a scope easily manageable by existing diffusion models.
2. **Comprehensive and scalable data construction strategy**: The combination of piece-wise Gaussian reconstruction, extrapolation rendering, and Gaussian perturbation efficiently constructs 1.5 million training samples with controllable costs.
3. **Generalizable progressive strategy**: The progressive expansion concept from small to large offsets is not only suitable for driving simulation but can also be generalized to other scene reconstruction tasks that require extrapolation rendering.
4. **Significant practical value**: Free-viewpoint simulation is a mandatory capability for truly practical driving simulators. FreeSim takes a crucial step toward this goal.

## Limitations & Future Work

1. There is a slight drop in rendering quality on the recorded trajectory (PSNR drops from 29.19 to 28.32), as the generation process inevitably introduces minor inconsistencies.
2. Progressive reconstruction increases total training time, requiring additional rounds of generation-reconstruction for each scene.
3. The randomness of diffusion generation may cause inconsistencies in detailed textures across different viewpoints.
4. The rolling shutter distortion is mitigated through post-processing but is not fundamentally resolved.
5. The evaluation only selects 16 scenes; larger-scale verification is required in future work.

## Related Work & Insights

- **Comparison with SGD**: SGD relies on diffusion generation conditioned on reference images and depth maps, primarily handling rotation changes and struggling with large spatial translations. FreeSim overcomes this constraint through of a degradation-enhancement paradigm.
- **Comparison with FreeVS**: FreeVS uses LiDAR projections as pseudo-image conditions, which is constrained by LiDAR coverage. FreeSim utilizes LiDAR only as an auxiliary condition, with the degraded rendered image as the primary condition, providing more comprehensive coverage.
- **Insight on efficiency from piece-wise reconstruction**: The strategy of segmenting large scenes into small clips for reconstruction is highly efficient for data construction (achieving a 6x speedup). This methodology can be applied to prepocessing other large-scale scenes.
- **Complementary to World Models**: Compared to world model-based methods like DriveDreamer4D, FreeSim's reconstruction + enhancement approach might be more suitable for synthesizing unconventional trajectories (such as translating viewpoints upward, which represents motion patterns unseen by the world model).

## Rating

⭐⭐⭐⭐⭐ (5/5)

The problem addressed is both important and highly challenging. The core ideas (degraded enhancement + progressive expansion) are elegant and effective. The experimental results significantly outperform all baselines in large-offset scenarios, supported by thorough ablation studies. The overall completeness of the work is high, providing a strong impetus to the field of driving simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Generating Multimodal Driving Scenes via Next-Scene Prediction](generating_multimodal_driving_scenes_via_next-scene_prediction.md)
- [\[CVPR 2025\] Scenario Dreamer: Vectorized Latent Diffusion for Generating Driving Simulation Environments](scenario_dreamer_vectorized_latent_diffusion_for_generating_driving_simulation_e.md)
- [\[ICLR 2026\] SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms](../../ICLR2026/autonomous_driving/simuli_real-time_lidar_and_camera_simulation_with_unscented_transforms.md)
- [\[CVPR 2025\] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)
- [\[CVPR 2025\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)

</div>

<!-- RELATED:END -->
