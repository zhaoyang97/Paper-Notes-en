---
title: >-
  [Paper Note] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model
description: >-
  [CVPR 2025][Autonomous Driving][Traffic Simulation] SceneDiffuser++ is proposed, an end-to-end city-scale traffic simulation diffusion model that handles agent spawning and despawning in sparse tensors via soft clipping, achieving over 60 seconds of trip-level traffic simulation with a combined JS divergence of 0.2423 on WOMD-XLMap.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Traffic Simulation"
  - "Diffusion Models"
  - "City-Scale"
  - "Soft Clipping"
  - "Sparse Tensors"
date: 2026-05-08
content_hash: 313d76c6998c20f2
---

# SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model

**Conference**: CVPR 2025  
**arXiv**: [2506.21976](https://arxiv.org/abs/2506.21976)  
**Code**: None  
**Area**: Autonomous Driving / Traffic Simulation  
**Keywords**: Traffic Simulation, Diffusion Models, City-Scale, Soft Clipping, Sparse Tensors

## TL;DR

SceneDiffuser++ is proposed, an end-to-end city-scale traffic simulation diffusion model that handles agent spawning and despawning in sparse tensors via soft clipping, achieving over 60 seconds of trip-level traffic simulation with a combined JS divergence of 0.2423 on WOMD-XLMap.

## Background & Motivation

### Background

**Background**: Autonomous driving simulation requires realistic traffic scene generation. Existing methods either only support short-term simulation (<10s) or fail to handle agent spawning/despawning (entering/leaving the scene). City-scale long-term simulation requires simultaneous modeling of the behaviors of hundreds of agents, traffic lights, occlusions, and spawn/despawn dynamics.

**Limitations of Prior Work**: Sparse tensors present a core technical challenge: different agents appear/disappear at different times, leading to a large number of invalid slots in the data tensor. Traditional methods handle this using zero-padding or hard clipping, but this degrades generation quality during diffusion model inference.

**Key Challenge**: Diffusion models must operate on all positions simultaneously during denoising, yet valid and invalid positions in sparse tensors require differentiated treatment—valid positions must need precise denoising, while invalid positions must remain zero.

**Key Insight**: The model is allowed to simultaneously predict feature values and validity masks, enabling soft clipping with masks to smoothly zero out invalid positions during inference.

**Core Idea**: v-prediction diffusion + soft clipping for sparse agents + multi-tensor heterogeneous modeling = end-to-end city-scale traffic simulation.

## Method

### Key Designs

1. **Soft Clipping**:

    - **Function**: Elegantly handling valid/invalid positions of sparse tensors during diffusion inference
    - **Mechanism**: The model simultaneously predicts feature values and a validity mask $M(x)$. During inference, $\hat{x}_t \leftarrow V(\hat{x}_t) \cdot M(\hat{x}_t)$—positions with validity masks near 1 retain their feature values, while positions near 0 are set to zero. Compared to hard clipping (binary thresholding), soft clipping maintains differentiability
    - **Design Motivation**: Hard clipping introduces discontinuities in the intermediate steps during early denoising, leading to degraded generation quality

2. **Multi-Tensor Heterogeneous Scene Modeling**:

    - **Function**: Unifying the processing of scene elements with different dimensions
    - **Mechanism**: Agents (position/orientation/velocity = 5 dimensions) and traffic lights (status = 4 dimensions) are treated as distinct tensors, sharing the same diffusion model but with their own projection layers
    - **Design Motivation**: Traffic lights and agents have completely different feature dimensions and semantics; forcing them into the same dimension would result in information loss

### Loss & Training

v-prediction diffusion loss $L = \mathbb{E}[\|(\tilde{v}_\theta - v_t) \cdot w\|_2^2]$ with higher weights for valid positions. Training data: WOMD-XLMap (1km radius augmented map), 600-step (60-second) rollout, replanning every 40 steps.

## Key Experimental Results

### Main Results

WOMD-XLMap 60s simulation JS divergence ↓:

| Method | Combined Score | Traffic Light Violations | Valid Agents |
|------|--------|-----------|-----------|
| SceneDiffuser | ~0.29 | ~0.20 | ~0.35 |
| **SceneDiffuser++** | **0.2423** | **0.1625** | **0.3053** |

### Ablation Study

| Clipping Strategy | Combined Score |
|---------|--------|
| No Clipping | Poor |
| Hard Clipping | Fair |
| **Soft Clipping** | **Optimal** |

### Key Findings
- Soft clipping is key to diffusion generation on sparse tensors—hard clipping introduces artifacts during the denoising process.
- 40-step replanning is the optimal balance point between speed and quality.
- Ultra-long simulation (3000 steps / 5 minutes) remains feasible but suffers from performance degradation.

## Highlights & Insights
- **First end-to-end city-scale trip simulation**—a single model handles agent behavior, spawning/despawning, traffic lights, and occlusions.
- **Generality of soft clipping**—can be extended to any diffusion generation task on sparse tensors.

## Limitations & Future Work
- Simulation drift still exists after extremely long durations.
- Validated only on the WOMD dataset.
- Agents may run off the map in ultra-long simulations.

## Rating
- Novelty: ⭐⭐⭐⭐ Soft clipping and multi-tensor architecture resolve practical issues
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed evaluation across multiple metrics
- Writing Quality: ⭐⭐⭐⭐ Clear
- Value: ⭐⭐⭐⭐ Advances the practicality of city-scale traffic simulation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](generative_gaussian_splatting_for_unbounded_3d_city_generation.md)
- [\[CVPR 2025\] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)
- [\[CVPR 2026\] SimScale: Learning to Drive via Real-World Simulation at Scale](../../CVPR2026/autonomous_driving/simscale_learning_to_drive_via_real-world_simulation_at_scale.md)
- [\[CVPR 2025\] FreeSim: Toward Free-Viewpoint Camera Simulation in Driving Scenes](freesim_toward_free-viewpoint_camera_simulation_in_driving_scenes.md)
- [\[CVPR 2025\] MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction](maskgwm_a_generalizable_driving_world_model_with_video_mask_reconstruction.md)

</div>

<!-- RELATED:END -->
