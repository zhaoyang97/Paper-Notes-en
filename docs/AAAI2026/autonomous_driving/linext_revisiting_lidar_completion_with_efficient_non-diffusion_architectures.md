---
title: >-
  [Paper Note] LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures
description: >-
  [AAAI 2026][Autonomous Driving][LiDAR scene completion] This paper proposes LiNeXt, a lightweight non-diffusion network for LiDAR 3D scene completion. Through a **Distance-aware Selective Repetition (DSR) strategy**…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "LiDAR scene completion"
  - "point cloud completion"
  - "non-diffusion model"
  - "sparse convolution"
  - "real-time inference"
date: 2026-05-08
content_hash: 5dd9471a1011bfbb
---

# LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures

**Conference**: AAAI 2026
**arXiv**: [2511.10209](https://arxiv.org/abs/2511.10209)  
**Code**: N/A  
**Area**: Autonomous Driving
**Keywords**: LiDAR scene completion, point cloud completion, non-diffusion model, sparse convolution, real-time inference

## TL;DR

This paper proposes LiNeXt, a lightweight non-diffusion network for LiDAR 3D scene completion. Through a **Distance-aware Selective Repetition (DSR) strategy**, a **Noise-to-Coarse (N2C) module**, and a **Refine module**, LiNeXt directly reconstructs complete point clouds. On SemanticKITTI, it achieves **199.8×** faster inference than LiDiff, reduces Chamfer Distance by **50.7%**, and uses only **6.1%** of LiDiff's parameters.

## Background & Motivation

### Problem Definition
Autonomous driving perception systems rely on LiDAR for 3D point clouds, but LiDAR measurements are inherently sparse and subject to frequent occlusions, resulting in large unobserved regions. Scene completion aims to infer and reconstruct missing spatial structures from sparse point clouds, providing complete 3D representations for downstream tasks such as object detection, pose estimation, and mapping.

### Limitations of Prior Work

**Voxel/SDF methods**: Constrained by resolution trade-offs — low resolution fails to capture fine geometric detail, while high resolution incurs prohibitive memory and computational costs.

**Diffusion-based methods** (LiDiff, LiDPM, ScoreLiDAR):
   - Despite high generation quality, **multi-step iterative sampling** introduces enormous computational overhead (LiDiff: 33.4 s/frame).
   - High noise magnitudes cause severe point displacement, increasing the difficulty of noise estimation and removal.
   - Complex network architectures with large parameter counts (LiDiff: 32.67M; LiDiff†: 54.40M).

### Root Cause
- **Directly minimizing Chamfer Distance** is simpler and more effective than the denoising objective of diffusion models — why not reconstruct scenes directly with a lightweight network?
- LiDAR point clouds exhibit **distance-dependent spatial distributions** (dense near, sparse far); uniform replication of input points leads to imbalanced distributions.
- Key observation: point clouds directly encode complex geometry and fine spatial detail without the resolution compromises of voxelization.

## Method

### Overall Architecture

LiNeXt operates in three stages:
1. **Distance-aware Selective Repetition (DSR)**: Groups input points by distance and applies different repetition factors per group, producing a more uniformly distributed noisy point cloud.
2. **Noise-to-Coarse (N2C) Module**: Reconstructs a coarse scene structure from the noisy point cloud in a single forward pass.
3. **Refine Module**: Refines the coarse output using intermediate features from N2C.

### Key Designs

#### 1. Distance-aware Selective Repetition (DSR)

**Function**: Addresses the near-dense/far-sparse imbalance caused by uniform replication in existing methods.

**Mechanism**: Points are sorted by distance and divided into four groups; near points are repeated fewer times and far points more:

Given input point cloud $P_{input} = \{p_i\}_{i=1}^N$, the distance from each point to the origin is $d_i = \|p_i\|$. Points are sorted in ascending order of distance and split into four equal groups $G_1, G_2, G_3, G_4$ with repetition counts $\{r_1=5, r_2=8, r_3=12, r_4=15\}$. Gaussian noise is added to the repeated point set to obtain $P_{noise}$.

**Design Motivation**: The physical characteristics of LiDAR result in dense near-field and sparse far-field sampling. Uniform replication over-samples the near field and under-samples the far field. DSR ensures uniform coverage across distances, providing a richer and more balanced input for the subsequent N2C module.

#### 2. Multi-Scale Sparse Convolution Module (MSSC)

**Function**: Extracts point cloud features in parallel at multiple voxel resolutions, capturing both fine local geometry and coarse global context.

**Mechanism**: Parallel sparse convolutions are applied at $N_{vox}$ voxel scales $g_k \in \{0.01 \times 2^{i-1} | i=1,...,N_{vox}\}$. For each scale:
- Voxelization: $\hat{P}_k = \lfloor P/g_k \rfloor$
- Feature encoding: $F_k = \text{MLP}_k(X)$
- Dual-residual sparse convolution: $\mathcal{T}_k'' = \text{spconv}_{k,2}(\mathcal{T}_k') + \mathcal{T}_k'$
- Multi-scale fusion: $F = \text{MLP}_{end}(\text{CONCAT}(O_k))$

**Design Motivation**: Different voxel resolutions capture spatial information at different levels — fine-grained scales preserve geometric detail while coarse scales capture global context. Residual connections maintain geometric fidelity.

#### 3. Cross-Point Attention Module (CPA)

**Function**: Performs robust feature fusion between global scene and local partial representations, strengthening inference of missing structures by explicitly encoding spatial relationships.

**Mechanism**: Given primary point coordinates $P_{key}$ and partial coordinates $P_{query}$ (with query and value features):

1. **KNN search** establishes local correspondences: $idx = \text{KNN}(P_{query}, P_{key}, k)$
2. **Spatial embedding** computes relative displacement: $\alpha = \text{MLP}_{pos}(P_{key} - \mathcal{G}(P_{key}, idx))$
3. **Relational features** enhance geometry-aware differences: $Q_{rel} = query - \mathcal{G}(key, idx) + \alpha$
4. **Serialized Segmented Max Pooling (SSMP)**: Divides the neighborhood into $\hat{K}$ segments and applies max pooling per segment for key-dimension compression.
5. **Attention aggregation**: $\mathcal{A} = \text{SoftMax}(\text{MLP}_{attn}(\hat{Q}_{rel}))$, followed by weighted aggregation.

**Design Motivation**: Standard cross-attention has quadratic complexity that is infeasible at point cloud scale under a 24 GB memory constraint. CPA compresses dimensionality via segmented pooling while retaining discriminative patterns. Z-order and Hilbert-order serialization preserve spatial locality.

#### 4. Noise-to-Coarse (N2C) Module

**Function**: Generates a coarse denoised point cloud directly from the noisy input through a hierarchical structure-distillation architecture.

**Core Pipeline**:
1. **Initial feature extraction**: MSSC extracts features $F_0, F_{noise}$ from the input and noisy point clouds respectively.
2. **Hierarchical seed generation**: $N$-stage FPS downsampling + CPA progressive refinement → outputs global seeds $P_{seed}, F_{seed}$.
3. **Coarse reconstruction**: PointNet fuses noisy coordinates with seed features; the final CPA module regresses coarse coordinates and features.

#### 5. Refine Module

**Function**: Recovers structural completeness and geometric detail accuracy from the coarse output.

**Mechanism**: Retrieves regional features from the seed set → PointNet generates relation-aware values → CPA refinement → deconvolution upsampling. This constitutes the second level of a coarse-to-fine pipeline.

### Loss & Training

The model is trained directly with Chamfer Distance; ground-truth point clouds are downsampled to 180,000 points:

$$L_{CD}(P, \hat{P}) = \frac{1}{|P|}\sum_{x \in P}\min_{y \in \hat{P}}\|x-y\|_2^2 + \frac{1}{|\hat{P}|}\sum_{y \in \hat{P}}\min_{x \in P}\|y-x\|_2^2$$

The Coarse and Refine stages are trained independently. Training is conducted on SemanticKITTI only, without fine-tuning on KITTI-360.

## Key Experimental Results

### Main Results

**SemanticKITTI dataset:**

| Method | CD↓ | JSD 3D↓ | JSD BEV↓ | IoU(0.5m)↑ | IoU(0.2m)↑ | IoU(0.1m)↑ |
|---|---|---|---|---|---|---|
| LMSCNet | 0.641 | - | 0.431 | 30.83 | 12.09 | 3.65 |
| LiDiff | 0.434 | 0.564 | 0.444 | 31.47 | 16.79 | 4.67 |
| LiDPM | 0.446 | 0.532 | 0.440 | 34.09 | 19.45 | 6.27 |
| ScoreLiDAR | 0.406 | - | 0.425 | - | - | - |
| **LiNeXt** | **0.214** | **0.494** | **0.336** | **41.07** | 19.45 | 6.30 |
| LiDiff† | 0.376 | 0.573 | 0.416 | 32.43 | 22.99 | 13.40 |
| LiDPM† | 0.376 | 0.542 | 0.403 | 36.59 | 25.76 | 14.93 |
| **LiNeXt†** | **0.149** | **0.481** | **0.331** | **41.97** | **31.25** | **15.09** |

**Efficiency comparison:**

| Method | CD↓ | Inference Time (s) | Params (M) |
|---|---|---|---|
| LiDiff | 0.434 | 33.359 | 32.67 |
| LiDPM | 0.446 | 15.288 | 32.67 |
| ScoreLiDAR | 0.406 | 5.047 | 32.67 |
| **LiNeXt** | **0.214** | **0.167** | **1.99** |
| LiDiff† | 0.376 | 33.531 | 54.40 |
| **LiNeXt†** | **0.149** | **0.434** | **2.10** |

Key figures: compared with LiDiff, LiNeXt achieves **199.8× speedup**, **50.7% reduction in CD**, and uses only **6.1% of the parameters**.

### Ablation Study

| Configuration | CD↓ | JSD 3D↓ | JSD BEV↓ | IoU(0.5m)↑ | Note |
|---|---|---|---|---|---|
| **LiNeXt** | **0.214** | **0.494** | **0.336** | **41.07** | Full model |
| w/o DSR | 0.215 | 0.508 | 0.352 | 40.00 | Degraded global shape consistency |
| w/o MSSC | 0.221 | 0.502 | 0.350 | 39.87 | Reduced fine structure recovery |
| w/o CPA | 0.227 | 0.504 | 0.353 | 39.36 | Largest drop; hierarchical feature aggregation is critical |

### Key Findings

1. **Non-diffusion methods can substantially outperform diffusion methods**: LiNeXt is not only faster but also achieves higher completion quality (CD: 0.434 → 0.214), challenging the assumption that diffusion models yield superior quality.
2. **Strong cross-dataset generalization**: A model trained solely on SemanticKITTI and tested directly on KITTI-360 maintains CD = 0.149, whereas LiDiff† degrades from 0.376 to 0.517.
3. **CPA is the most critical module**: Removing CPA causes the largest performance drop (CD +6.1%), demonstrating that hierarchical feature aggregation is essential for scene completion.
4. **DSR has global impact**: Although the CD change is small, JSD and IoU metrics degrade noticeably without DSR, indicating that uniform distribution is important for global shape consistency.

## Highlights & Insights

1. **Challenges the necessity of diffusion models for generative tasks**: In LiDAR completion, direct regression not only runs faster but also achieves higher quality, suggesting that diffusion models are not the optimal choice for certain structured tasks.
2. **Distance-aware design** aligns with the physical characteristics of LiDAR: the "fewer near, more far" repetition strategy is simple yet effective, improving performance at negligible cost.
3. **Serialized Segmented Max Pooling (SSMP)** is an elegant attention compression technique that substantially reduces computational complexity while preserving discriminative patterns.
4. The extremely lightweight design of **1.99M parameters** is well-suited for embedded deployment, which is highly relevant for real-world autonomous driving applications.

## Limitations & Future Work

1. In ablation experiments, CPA can only be partially replaced in the Refine module (full replacement exceeds 24 GB memory); complete ablation results may differ.
2. Evaluation is limited to SemanticKITTI and KITTI-360; larger-scale datasets (e.g., Waymo, nuScenes) are not covered.
3. Only static scenes are addressed; dynamic objects are not considered.
4. The advantage of IoU at fine granularity (0.1 m) is marginal, indicating room for improvement in fine structure recovery.

## Related Work & Insights

- **LiDiff** (diffusion baseline): locally guided diffusion under the DDPM framework; serves as the primary comparison target.
- **ScoreLiDAR**: accelerates diffusion sampling 5× via knowledge distillation, yet remains far slower than LiNeXt.
- **SnowflakeNet**: deconvolution-based upsampling for point cloud completion, adopted in LiNeXt's Refine module.
- Insight: "appropriate inductive bias + lightweight design" often outperforms "generic but heavy generative models."

## Rating

- Novelty: ⭐⭐⭐⭐ — The initiative to challenge the diffusion paradigm is commendable; DSR and CPA are novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual-dataset evaluation, cross-dataset generalization, efficiency comparison, and thorough ablation study.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, structure is well-organized, and Figure 1 is highly intuitive.
- Value: ⭐⭐⭐⭐⭐ — The 199.8× speedup is highly significant for practical deployment; the practical value is exceptional.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](../../ICCV2025/autonomous_driving/distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)
- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](../../CVPR2026/autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [\[AAAI 2026\] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)
- [\[AAAI 2026\] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [\[AAAI 2026\] Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models](unlocking_efficient_vehicle_dynamics_modeling_via_analytic_world_models.md)

</div>

<!-- RELATED:END -->
