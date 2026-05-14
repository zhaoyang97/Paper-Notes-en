---
title: >-
  [Paper Note] 4D Visual Pre-training for Robot Learning
description: >-
  [ICCV 2025][3D Vision][Point cloud pre-training] FVP formulates 3D visual pre-training as a *next-point-cloud-prediction* problem, training a conditional diffusion model to predict the current-frame point cloud from hist…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Point cloud pre-training"
  - "diffusion models"
  - "imitation learning"
  - "3D representation learning"
  - "robot manipulation"
date: 2026-05-08
content_hash: 40a499f276c05691
---

# 4D Visual Pre-training for Robot Learning

**Conference**: ICCV 2025
**arXiv**: [2508.17230](https://arxiv.org/abs/2508.17230)
**Code**: [https://github.com/JackHck/FVP](https://github.com/JackHck/FVP)
**Area**: 3D Vision / Robot Learning
**Keywords**: Point cloud pre-training, diffusion models, imitation learning, 3D representation learning, robot manipulation

## TL;DR
FVP formulates 3D visual pre-training as a *next-point-cloud-prediction* problem, training a conditional diffusion model to predict the current-frame point cloud from historical-frame point clouds. This approach achieves a 28% average success rate improvement over DP3 across 12 real-world manipulation tasks, establishing a new state of the art.

## Background & Motivation
- Existing robot visual representation learning predominantly relies on 2D image pre-training (e.g., R3M, MVP, VC-1), neglecting the inherently 3D nature of the physical world.
- 3D point clouds have demonstrated superior efficiency and generalization over 2D images as visual inputs for robot manipulation (e.g., DP3, RISE).
- However, large-scale 3D data is scarce on the internet, making it difficult to train general-purpose 3D representations from web data as done in the 2D domain.
- Existing 3D pre-training methods (contrastive learning, masked reconstruction) do not sufficiently exploit temporal information for understanding robot motion dynamics.

## Core Problem
1. **How can a general-purpose 3D visual pre-training method be designed to improve robot manipulation performance in the absence of large-scale 3D data?**
2. **How should the pre-training objective be designed so that the visual encoder learns the dynamic changes of the physical environment and robot motion characteristics?**
3. **Can the pre-training framework generalize across different 3D encoders, datasets, robot platforms (single-arm / dual-arm / humanoid), and large VLA models?**

## Method

### Overall Architecture
FVP (4D Visual Pre-training) formulates the visual pre-training objective as **next-point-cloud-prediction**. The pipeline consists of two stages:
1. **Pre-training stage**: A 3D visual encoder maps the previous-frame point cloud $o_{t-1}$ to a latent representation $\mathbf{z}$, which then conditions a point cloud diffusion model to iteratively denoise Gaussian noise into the current-frame point cloud $o_t$.
2. **Downstream fine-tuning**: The pre-trained 3D visual encoder replaces the encoder in a downstream imitation learning method (e.g., DP3, RISE) and is fine-tuned end-to-end.

### Key Designs
1. **Next-Point-Cloud-Prediction objective**: Unlike contrastive learning (same timestep as positive pairs, different timesteps as negatives) or masked point cloud reconstruction, FVP predicts the next-frame point cloud conditioned on the previous-frame observation. This enables the visual model to capture robot motion characteristics and temporal environment dynamics, acquiring the behavioral information critical for imitation learning.
2. **Conditional diffusion model**: A Point-Voxel Diffusion network serves as the denoising backbone. The latent representation $\mathbf{z} \in \mathbb{R}^{N \times C_v}$ output by the visual encoder is concatenated with the noisy point cloud $o_t^T \in \mathbb{R}^{N \times 3}$ to form $o_t^{T,+} \in \mathbb{R}^{N \times (C_v+3)}$, from which the diffusion model predicts the noise $\epsilon$.
3. **Universal encoder interface**: FVP imposes no constraints on the 3D encoder architecture, supporting PointNet++, Point Transformer, DP3 Encoder, RISE Encoder, and others, making it a plug-and-play pre-training module.
4. **Flexible pre-training data**: FVP supports both in-domain (small-scale task demonstration data) and out-of-domain (e.g., the large-scale RoboMind dataset) pre-training.

### Loss & Training
- **Pre-training loss**: Standard $L_2$ noise prediction loss of the diffusion model:
  $$\mathcal{L} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \mathbf{I})} \left[ \| \epsilon - \epsilon_\theta(o_t^{T,+}, T) \|_2^2 \right]$$
- **Downstream fine-tuning**: The pre-trained encoder initializes the encoder of DP3/RISE and is fine-tuned end-to-end (unfrozen). Ablation experiments show that freezing the encoder significantly degrades performance due to domain gap.
- Using a single previous frame (1 frame) as the conditioning input yields the best results; incorporating additional historical frames proves detrimental.

## Key Experimental Results

### Simulation Experiments (Adroit + MetaWorld)

| Method | In-domain Avg. Gain | Out-of-domain Avg. Gain |
|---|---|---|
| FVP (DP3) | +16.9% | +24.7% |

FVP surpasses 3D pre-training methods including PointMAE, STRL, and C2P, as well as 2D pre-training methods R3M and MVP, on both Adroit and MetaWorld benchmarks.

### Real-World Experiments (12 Tasks, 4 Robot Types)

| Task | DP3 | DP3+FVP | RISE+FVP |
|---|---|---|---|
| PickSquare | 14/20 | 20/20 | 20/20 |
| PlaceBottle | 13/20 | 20/20 | 19/20 |
| PickPlace | 11/20 | 17/20 | 17/20 |
| FlipCup | 10/20 | 16/20 | 14/20 |
| Assembly | 6/20 | 13/20 | 13/20 |
| ArtiManip | 7/20 | 16/20 | 13/20 |

FVP achieves absolute success rate improvements of 15%–55% across real-world tasks.

### 2D Pre-training vs. FVP (using DP3 as the policy)

| Method | Average |
|---|---|
| R3M | 12.5/20 |
| MVP | 15.5/20 |
| MAE (Soup-1M+100 DoH) | 15.3/20 |
| DP3+FVP | **16.4/20** |

### VLA Model (RDT-1B) Results

| Input | PickSquare | PlaceBottle | PutBox | StackBowl | WipePlate |
|---|---|---|---|---|---|
| 2D Image | 12/20 | 10/20 | 6/20 | 8/20 | 3/20 |
| 2D Image + R3M | 15/20 | 12/20 | 7/20 | 11/20 | 4/20 |
| 3D Point Cloud | 14/20 | 12/20 | 9/20 | 13/20 | 4/20 |
| 3D + FVP Pre-training | **18/20** | **17/20** | **9/20** | **16/20** | **5/20** |

### Ablation Study
1. **Historical-frame vs. current-frame conditioning**: Using the previous frame as the condition significantly outperforms using the current frame (e.g., PickSquare: 20/20 vs. 15/20), confirming the importance of temporal information for pre-training.
2. **Frozen vs. fine-tuned encoder**: Freezing the pre-trained encoder causes a sharp performance drop on downstream tasks (e.g., PickSquare: 20/20 → 11/20), indicating that domain gap between out-of-domain pre-training and in-domain tasks necessitates end-to-end fine-tuning.
3. **Number of historical frames**: A single frame yields the best results; performance degrades monotonically as more frames are added (PickSquare: 20→19→17→15 for 1/2/3/4 frames), suggesting that excessive historical context introduces noise.

## Highlights & Insights
1. **Novel pre-training paradigm**: FVP is the first to adopt next-point-cloud-prediction as a 3D visual pre-training objective, naturally incorporating temporal dynamics in a manner more suitable for robot tasks than contrastive learning or masked reconstruction.
2. **Strong generality**: FVP supports arbitrary 3D encoders (PointNet++, Point Transformer, DP3 Encoder) and is compatible with diverse robot platforms (single-arm with gripper/dexterous hand, dual-arm, humanoid), and extends to large VLA models.
3. **Large-scale real-world validation**: Consistent and substantial gains are demonstrated across 12 real-world manipulation tasks and 4 robot morphologies.
4. **Simplicity and effectiveness**: The core idea is intuitive and straightforward to implement; as a plug-and-play module, FVP directly enhances existing 3D imitation learning methods.

## Limitations & Future Work
1. **Dependence on point cloud data**: The method requires datasets with depth or point cloud information. Large-scale open-source datasets such as Open-X-Embodiment lack camera parameters and depth data, precluding their direct use.
2. **Limited pre-training data scale**: Due to the scarcity of 3D data, pre-training is currently conducted on small in-domain datasets or RoboMind; the effectiveness of truly web-scale large-scale pre-training has not yet been validated.
3. **End-to-end fine-tuning required**: The significant performance drop when the encoder is frozen indicates that the pre-trained representations are not fully domain-agnostic, and domain gap remains a concern.
4. **Limited gains on VLA models**: Improvements on language-conditioned and long-horizon tasks are relatively modest (e.g., long-horizon tasks improve from only 0/20 to 3/20), suggesting that 3D information provides limited benefit for high-level semantic reasoning.

## Related Work & Insights

| Dimension | FVP | DP3 | RISE |
|---|---|---|---|
| Pre-training | ✅ Next-point-cloud-prediction | ❌ None | ❌ None |
| Input | 3D point cloud | 3D point cloud | 3D point cloud |
| Encoder | Universal (multiple supported) | Lightweight DP3 Encoder | Sparse conv + Transformer |
| Role | Pre-training module | End-to-end policy | End-to-end policy |

- **vs. PointMAE / STRL / C2P** (3D pre-training methods): These methods rely on masked reconstruction or contrastive learning without exploiting temporal information. FVP introduces dynamic information by predicting the next frame, yielding substantial improvements in both simulation and real-world tasks.
- **vs. R3M / MVP** (2D pre-training methods): Despite being pre-trained on far larger datasets (>300M samples), 2D representations are less effective than FVP's 3D pre-trained representations for 3D manipulation tasks.

### Broader Implications
1. **Universality of the "next-X-prediction" paradigm**: Next-token-prediction in LLMs, next-frame-prediction in video models, and next-point-cloud-prediction here all demonstrate that predicting the next temporal state is a highly effective self-supervised objective across modalities.
2. **3D vs. 2D for robot manipulation**: This work further substantiates the advantage of 3D point cloud inputs for robot manipulation, particularly in scenarios requiring fine-grained 3D spatial perception such as dexterous hand manipulation.
3. **Pre-training + fine-tuning in robotics**: With the emergence of 3D robot datasets such as RoboMind, large-scale 3D pre-training is poised to become a new trend.
4. **Extensibility to other 3D tasks**: The next-point-cloud-prediction objective may also benefit other domains requiring dynamic 3D perception, such as autonomous driving and 3D scene understanding.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing the next-prediction paradigm to 3D point cloud pre-training is a novel contribution, though the diffusion model itself is an existing tool.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 real-world tasks, 4 robot types, simulation and real-world evaluation, multiple encoders, VLA extension, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-organized experiments; some notation and descriptions could be made more concise.
- Value: ⭐⭐⭐⭐⭐ A simple yet effective general-purpose method with large-scale real-world validation; of significant reference value to the 3D robot learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[ICCV 2025\] RoboPearls: Editable Video Simulation for Robot Manipulation](robopearls_editable_video_simulation_for_robot_manipulation.md)
- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](../../CVPR2026/3d_vision/e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)

</div>

<!-- RELATED:END -->
