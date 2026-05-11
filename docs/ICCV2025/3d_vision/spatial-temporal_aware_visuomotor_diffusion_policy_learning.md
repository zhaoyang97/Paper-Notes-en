---
title: >-
  [Paper Note] Spatial-Temporal Aware Visuomotor Diffusion Policy Learning
description: >-
  [ICCV 2025][3D Vision][Visual Imitation Learning] This paper proposes the 4D Diffusion Policy (DP4), which injects 3D spatial and 4D spatial-temporal awareness into a diffusion policy via a dynamic Gaussian world model…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Visual Imitation Learning"
  - "Diffusion Policy"
  - "3D Gaussian Splatting"
  - "4D Spatial-Temporal Awareness"
  - "World Model"
  - "Dexterous Manipulation"
date: 2026-05-08
content_hash: 46ee1d9511bbac17
---

# Spatial-Temporal Aware Visuomotor Diffusion Policy Learning

**Conference**: ICCV 2025
**arXiv**: [2507.06710](https://arxiv.org/abs/2507.06710)
**Code**: [Project Page](https://ZhenyangLiu.github.io/DP4)
**Area**: 3D Vision / Robot Manipulation
**Keywords**: Visual Imitation Learning, Diffusion Policy, 3D Gaussian Splatting, 4D Spatial-Temporal Awareness, World Model, Dexterous Manipulation

## TL;DR

This paper proposes the 4D Diffusion Policy (DP4), which injects 3D spatial and 4D spatial-temporal awareness into a diffusion policy via a dynamic Gaussian world model, achieving substantial improvements over baselines across 17 simulation tasks and 3 real-robot tasks (Adroit +16.4%, DexArt +14%, RLBench +6.45%, real tasks +8.6%).

## Background & Motivation

Visual Imitation Learning is an effective paradigm for training robots to perform complex tasks, with notable progress in object grasping, legged locomotion, and dexterous manipulation. However, existing methods suffer from two fundamental bottlenecks:

**Reliance on Behavior Cloning**: Mainstream methods perform behavior cloning by supervising historical trajectories, passively imitating expert actions without genuinely understanding the physical structure of the environment.

**Lack of Spatial-Temporal Awareness**: Such methods can neither accurately capture the 3D spatial structure of the current scene (e.g., precise object positions and geometry) nor model the 4D spatial-temporal dynamics during interaction (e.g., object motion tendencies).

This makes conventional approaches prone to failure when scenes change or precise object interaction is required. For instance, in a hammer-nail task, imitating the hammering motion without understanding the 3D position of the nail leads to frequent misses.

**Core Problem**: How can a diffusion policy model learn not only *how to move*, but also *what the environment looks like* and *how it will change*?

DP4 addresses this by constructing a **Dynamic Gaussian World Model** that explicitly encodes 3D spatial structure and 4D spatial-temporal dynamics into the policy learning process.

## Method

### Overall Architecture

The DP4 pipeline proceeds as follows:

1. **Input**: Single-view RGB-D images
2. **3D Point Cloud Construction**: RGB-D images are lifted into 3D point clouds
3. **Multi-level Feature Extraction**: Local and global 3D representations are extracted
4. **Diffusion Policy**: Conditioned on multi-level 3D representations and robot state, a conditional diffusion model generates action trajectories
5. **Gaussian World Model** *(training only)*: Constructs a 3DGS reconstruction of the current scene from point clouds → predicts future scenes → provides 3D/4D supervision

Key insight: The Gaussian world model participates only during training, introducing no additional inference overhead.

### Key Designs 1: Multi-level 3D Spatial Awareness

**Local 3D Representation**:
- The point cloud is cropped to the region of interest (e.g., near the robot arm) and downsampled to 512 points via Farthest Point Sampling (FPS)
- Encoded into compact local features via a lightweight MLP (3D Local Encoder)

**Global 3D Representation**:
- The full point cloud is voxelized into a $100^3$ voxel grid
- Encoded by a 3D convolutional encoder (3D Global Encoder, U-Net architecture) to produce global features $v^{(t)} \in \mathbb{R}^{100^3 \times 64}$

**Generalizable Gaussian Regressor**:
- Regresses Gaussian primitive parameters $\theta^{(t)} = (\mu, c, r, s, \sigma)$ (position, color, rotation, scale, opacity) from global voxel features
- Renders RGB and Depth images via differentiable tile-based rasterization
- Enhances spatial awareness through a 3D reconstruction loss:

$$\mathcal{L}_{3D} = \sum_{\mathbf{p}} \|\mathbf{C}^{(t)}(\mathbf{p}) - \mathbf{C}^{*(t)}\|_2^2 + \|\mathbf{D}^{(t)}(\mathbf{p}) - \mathbf{D}^{*(t)}\|_2^2$$

### Key Designs 2: 4D Spatial-Temporal Awareness

Temporal modeling is layered on top of the 3D Gaussian world model:

- **Deformable MLP**: Given current Gaussian parameters $\theta^{(t)}$ and action $a^{(t)}$, predicts parameter residuals $\Delta\theta^{(t)}$
- **Future Scene Reconstruction**: $\theta^{(t+1)} = \theta^{(t)} + \Delta\theta^{(t)}$, followed by rendering the RGB and Depth of the future timestep
- **4D Consistency Supervision**: Aligns predicted future scenes with ground-truth future observations:

$$\mathcal{L}_{4D} = \sum_{\mathbf{p}} \|\mathbf{C}^{(t+1)}(\mathbf{p}) - \mathbf{C}^{*(t+1)}\|_2^2 + \|\mathbf{D}^{(t+1)}(\mathbf{p}) - \mathbf{D}^{*(t+1)}\|_2^2$$

This compels the 3D representations to encode not only the current scene structure but also the implicit dynamics of the physical world.

### Key Designs 3: Diffusion Decision Module

A convolution-based diffusion policy with DDIM noise scheduling is adopted:

- Conditioned on multi-level 3D representations $r$ and robot state $q$
- Generates clean actions $a^0$ from Gaussian noise $a^K$ via $K$ denoising steps
- During training: 2 observation steps are predicted, 4 action steps are generated, and the last 3 are executed

### Loss & Training

The total loss is a weighted combination of three terms:

$$\mathcal{L}_{DP4} = \mathcal{L}_{action} + \lambda_{3D} \mathcal{L}_{3D} + \lambda_{4D} \mathcal{L}_{4D}$$

where $\lambda_{3D} = 0.1$ and $\lambda_{4D} = 0.01$. Training strategy: the Deformable MLP is frozen for the first 500 steps (warm-up), after which all modules are trained jointly.

## Key Experimental Results

### Main Results: Simulation Task Success Rate (%)

**Adroit Simulation (Dexterous Hand Manipulation)**:

| Method | Hammer | Door | Pen | Avg. |
|--------|--------|------|-----|------|
| IBC | 0 | 0 | 9 | 3.0 |
| BCRNN | 0 | 0 | 9 | 3.0 |
| Diffusion Policy | 48 | 50 | 25 | 41.0 |
| DP3 | 100 | 62 | 43 | 68.3 |
| **DP4 (Ours)** | **100** | **80** | **75** | **84.7** |

**DexArt Simulation (Articulated Object Manipulation)**:

| Method | Laptop | Faucet | Bucket | Toilet | Avg. |
|--------|--------|--------|--------|--------|------|
| DP | 69 | 23 | 58 | 46 | 49.0 |
| DP3 | 83 | 63 | 82 | 46 | 68.5 |
| **DP4 (Ours)** | **92** | **84** | **90** | **64** | **82.5** |

**RLBench Simulation (Multi-task Manipulation, 10 tasks, 166 variants)**:

| Method | Group 1 Avg. (5 tasks) | Group 2 Avg. (5 tasks) |
|--------|------------------------|------------------------|
| PerAct | 30.4 | 10.4 |
| GNFactor | 47.5 | 16.0 |
| ManiGaussian | 57.1 | 33.2 |
| **DP4 (Ours)** | **63.3** | **39.9** |

### Ablation Study (Adroit)

| RGB Supervision | Depth Supervision | 4D Dynamics | Hammer | Door | Pen |
|-----------------|-------------------|-------------|--------|------|-----|
| ✗ | ✗ | ✗ | 94.0 | 64.0 | 45.0 |
| ✔ | ✗ | ✗ | 96.0 | 68.0 | 48.0 |
| ✔ | ✔ | ✗ | 98.0 | 75.0 | 72.0 |
| ✔ | ✔ | ✔ | **100.0** | **80.0** | **75.0** |

### Real-Robot Experiments

| Method | Grasp Bottle | Stack Cups | Pour Water | Avg. |
|--------|--------------|------------|------------|------|
| DP | 36.0 | 44.0 | 28.0 | 36.0 |
| DP3 | 42.0 | 62.0 | 34.0 | 46.0 |
| **DP4 (Ours)** | **48.0** | **72.0** | **44.0** | **54.6** |

### Key Findings

1. **4D supervision contributes most**: Ablation results show that the success rate on the Pen task jumps from 47% to 75% upon adding 4D supervision, demonstrating the critical role of dynamic awareness in fine-grained manipulation.
2. **No additional inference overhead**: The Gaussian world model participates only during training; inference time increases by approximately 0.1 seconds (e.g., Hammer: 6.40s → 6.57s).
3. **Hyperparameter sensitivity**: $\lambda_{3D}=0.1$ and $\lambda_{4D}=0.01$ are the optimal configuration; deviations in either direction degrade performance.
4. **Single-view sufficiency**: The method requires no multi-camera setup and completes all tasks with a single RGB-D camera.

## Highlights & Insights

1. **World model as a training auxiliary, not an inference dependency**: This is an elegant design — the Gaussian world model provides additional supervision signals exclusively during training and is entirely discarded at inference, reaping the benefits of structured supervision without increasing deployment cost.
2. **From passive imitation to active understanding**: Conventional BC focuses solely on action correctness at the output layer, whereas DP4 additionally requires intermediate representations to reconstruct the 3D scene and predict future scenes, forcing the features to encode physically meaningful information.
3. **Hierarchical 3D + 4D design**: The 3D loss ensures spatial structure awareness, and the 4D loss layers temporal dynamics on top — a well-motivated, progressive design.
4. **Generalizable Gaussian Regressor**: Gaussian parameters are regressed directly from voxel features, circumventing the prohibitive cost of per-scene 3DGS optimization and enabling generalization across scenes.
5. **Comprehensive experimental coverage**: 17 simulation tasks + 173 variants + 3 real-robot tasks, spanning dexterous hands, articulated objects, and deformable objects.

## Limitations & Future Work

1. **Single-view constraint**: Only a single RGB-D camera is used, resulting in limited rendering quality (the paper acknowledges a lack of fine rendering detail); the approach may fail in heavily occluded scenes.
2. **Real-task success rates have room for improvement**: The best real-task success rate is 72% (stack cups), while pour water achieves only 44%, leaving a gap before practical deployment.
3. **Single-step future prediction**: The 4D supervision models only the $t \to t+1$ transition, without considering longer-horizon dynamic prediction.
4. **Depth dependency**: RGB-D input is required rather than pure RGB, limiting applicability in purely vision-based camera setups.
5. **Training cost**: Training requires an H100 80GB GPU for 3,000 epochs, placing considerable demands on computational resources.

## Related Work & Insights

- **3D Diffusion Policy (DP3)**: The direct predecessor of DP4, which employs point cloud 3D representations but lacks spatial-temporal awareness; DP4 augments it with a Gaussian world model.
- **ManiGaussian**: Also uses Gaussian splatting for robot manipulation, but as a perception module within a PerAct-style architecture; DP4 integrates it into a diffusion policy framework.
- **GNFactor**: Uses NeRF as a generalizable factor; DP4 replaces it with 3DGS for more efficient rendering and better scalability.
- **Diffusion Policy**: The foundational framework; DP4 retains the advantages of the diffusion policy while introducing structured 3D/4D supervision.
- **World Model line of work**: The Dreamer series performs future prediction in latent space; DP4 opts for prediction in explicit 3D representation space, yielding stronger physical interpretability.

**Insight**: Treating the world model as an auxiliary training supervisor rather than an inference component is a generalizable paradigm — it enables expensive structured knowledge to be effectively "distilled" into feature representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of using a 3DGS world model as training supervision for a diffusion policy is highly novel.
- **Technical Depth**: ⭐⭐⭐⭐ — The hierarchical 3D/4D supervision design is well-motivated; deformable Gaussian field prediction is technically non-trivial.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 17 simulation + 3 real-robot tasks, thorough ablations, and rich visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, well-articulated motivation.
- **Value**: ⭐⭐⭐⭐ — No additional inference overhead, single-view operation, strong practical deployment potential.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A solid contribution that effectively integrates 3D vision and world models into robot policy learning.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](7dgs_unified_spatial-temporal-angular_gaussian_splatting.md)
- [\[NeurIPS 2025\] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](../../NeurIPS2025/3d_vision/trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [\[CVPR 2026\] Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics](../../CVPR2026/3d_vision/efficient_hybrid_se3-equivariant_visuomotor_flow_policy_via_spherical_harmonics_.md)

</div>

<!-- RELATED:END -->
