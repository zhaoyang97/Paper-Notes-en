---
title: >-
  [Paper Note] Efficient Equivariant Transformer for Self-Driving Agent Modeling
description: >-
  [CVPR 2026][Autonomous Driving][SE(2)-equivariance] This paper proposes DriveGATr, an equivariant Transformer architecture based on 2D Projective Geometric Algebra (PGA) that achieves SE(2)-equivariance without explicit pairwise relative position encoding (RPE), attaining state-of-the-art performance on traffic simulation tasks while substantially reducing computational cost.
tags:
  - CVPR 2026
  - Autonomous Driving
  - SE(2)-equivariance
  - Geometric Algebra
  - Transformer
  - Traffic Simulation
date: 2026-05-08
content_hash: 0bf7c7acb653462f
---

# Efficient Equivariant Transformer for Self-Driving Agent Modeling

**Conference**: CVPR 2026
**arXiv**: [2604.01466](https://arxiv.org/abs/2604.01466)
**Code**: N/A
**Area**: Autonomous Driving
**Keywords**: SE(2)-equivariance, Geometric Algebra, Transformer, Traffic Simulation, Autonomous Driving

## TL;DR

This paper proposes DriveGATr, an equivariant Transformer architecture based on 2D Projective Geometric Algebra (PGA) that achieves SE(2)-equivariance without explicit pairwise relative position encoding (RPE), attaining state-of-the-art performance on traffic simulation tasks while substantially reducing computational cost.

## Background & Motivation

Modeling agent behavior in traffic scenes is a critical task for autonomous driving. The task exhibits natural SE(2) symmetry: applying any 2D rotation and translation to the entire scene should produce correspondingly transformed outputs for each agent.

The dominant approach for achieving SE(2)-equivariance is **explicit pairwise relative position encoding (RPE)**: computing the relative pose for every pair of agents or map elements and embedding it into the attention mechanism. This introduces $O(N^2)$ additional computational overhead, limiting scalability to larger scenes and batch sizes, and precluding the use of efficient attention kernels such as FlashAttention.

An alternative approach, DRoPE (2D Rotary PE), avoids scalability issues but lacks expressiveness (encoding no geometric information) and achieves only translation equivariance rather than full rotation equivariance.

## Method

### Overall Architecture

DriveGATr encodes all scene elements (agents and map nodes) as 8-dimensional **multivectors** in the 2D projective geometric algebra $\mathbb{R}^*_{2,0,1}$, and processes them through a series of equivariant Transformer blocks. The core innovation is that equivariant attention is realized via invariant inner products between multivectors, eliminating the need for explicit RPE and enabling standard dot-product attention.

The architecture consists of $N$ factorized attention blocks, each comprising:
- Agent–Map cross-attention (per timestep)
- Agent–Agent self-attention (per timestep)
- Temporal causal self-attention (per agent)
- Equivariant MLP
- Invariant adapter

### Key Designs

1. **Multivector Encoding**: A 2D pose $(x, y, \theta)$ is encoded as a single multivector in $\mathbb{R}^*_{2,0,1}$. Specifically, the bivector components encode the point $(x, y)$, while the vector components encode the oriented line passing through that point. Invariant features such as velocity and bounding box dimensions are encoded in auxiliary scalars. This representation supports SE(2) transformations—rotations and translations—via the sandwich product of the geometric product.

2. **Equivariant Network Primitives**:

    - **Linear layer**: Learns weights across projected components of each $k$-blade to guarantee equivariance.
    - **Geometric bilinear layer**: Enhances expressiveness via the geometric product and Join operator.
    - **Activation function**: GatedRELU, which gates the entire multivector using its scalar component.
    - **Normalization**: LayerNorm based on invariant inner products.
    - **Scaled dot-product attention**: Invariant inner products of multivectors, augmented with distance-aware expanded features, are concatenated and passed to standard dot-product attention.

3. **Distance-Aware Attention**: Additional invariant features $\phi(q), \psi(k)$ are computed from the query/key multivectors. When the bivector components represent points, $\phi(q) \cdot \psi(k)$ is proportional to the negative squared Euclidean distance between the two points. Concatenating these features to the standard Q/K vectors yields distance-sensitive attention.

4. **Invariant Adapter**: Agent actions are ultimately invariant quantities, yet the multivector features carry important geometric information. By transforming the global multivector features into each agent's local coordinate frame (an invariant operation) and then mapping them to auxiliary scalars via an MLP, the equivariant geometric information is effectively converted into an invariant representation suitable for action decoding.

### Loss & Training

- The action space is discretized via clustering (2048 action tokens per agent class).
- Cross-entropy loss is used to predict the next action.
- The 3M model uses 128-dimensional auxiliary features; the 30M model uses 512-dimensional features.
- Models are trained for 250K steps with a learning rate of $10^{-3}$ and cosine annealing.

## Key Experimental Results

### Main Results

| Method | Params | RMM ↑ | Kinematic ↑ | Interactive ↑ | Map-based ↑ | minADE ↓ |
|--------|--------|-------|-------------|---------------|-------------|----------|
| DriveGATr-30M | 30M | **0.7636** | 0.4890 | 0.7272 | 0.8120 | 1.3682 |
| SMART-7M | 7M | 0.7678 | 0.4894 | 0.7306 | 0.8163 | 1.3532 |
| BehaviorGPT | 3M | 0.7438 | 0.4254 | 0.7233 | 0.7976 | 1.3804 |
| Transformer+RPE | 3M | 0.7251 | 0.4708 | 0.6953 | 0.7808 | 1.7486 |
| DriveGATr-3M | 3M | 0.7620 | 0.4859 | 0.7264 | 0.8103 | 1.4192 |

### Ablation Study

| Configuration | RMM ↑ | minADE ↓ | Notes |
|---------------|-------|----------|-------|
| IA + DA | 0.7478 | 1.5798 | Base configuration |
| Map Attn k=4 | 0.7478 | 1.5798 | Attend to 4 nearest map tokens |
| Map Attn k=8 | 0.7528 | 1.5293 | Attend to 8 nearest |
| Map Attn All | **0.7617** | **1.4174** | Attend to all map tokens (best) |

### Key Findings

1. **DriveGATr-3M achieves the best performance among models of equal parameter count**: RMM is 2 percentage points higher than BehaviorGPT at the same scale and significantly outperforms all non-equivariant baselines. The 30M variant matches SMART-7M on realism metrics.

2. **Full map attention is critical**: Expanding agent map context from $k=4$ to all map tokens improves RMM by 1.4 percentage points and reduces minADE by 1.6. This is a key advantage of DriveGATr over RPE-based methods, which are constrained by memory to attend only to a small neighborhood.

3. **Substantial computational efficiency gains**: As the number of agents increases, the FLOP growth of DriveGATr is substantially slower than that of Transformer+RPE, since the latter's RPE computation introduces $O(N^2)$ additional overhead.

4. **Sample efficiency**: Benefiting from SE(2)-equivariance as an inductive bias, DriveGATr consistently outperforms non-equivariant methods across varying training set sizes (1%/10%/50%/100%).

5. **True rotation and translation invariance**: In experiments where the scene is rotated by 90° and translated by 100m, DriveGATr produces consistent trajectory predictions, whereas non-equivariant Transformers and the translation-only-equivariant DRoPE exhibit substantial prediction drift.

## Highlights & Insights

- The core contribution is adapting GATr (E(3)-equivariant) to a SE(2)-equivariant version for 2D driving scenarios, reducing the multivector dimension from 16 to 8 for improved computational efficiency.
- The design philosophy encodes symmetry naturally through mathematical structure (geometric algebra) rather than handcrafted relative position features, making equivariance a constructive guarantee rather than an approximation.
- The invariant adapter is an elegant design: it bridges equivariant features and invariant outputs by transforming features into each agent's local coordinate frame.
- Compatibility with efficient attention kernels such as FlashAttention is an important practical advantage for deployment.

## Limitations & Future Work

- Equivariance is only established in the 2D plane; real-world driving is a 3D problem (height could be incorporated via auxiliary scalars for a 2.5D extension).
- Evaluation is limited to traffic simulation; performance on related tasks such as motion forecasting and planning has not been validated.
- Techniques that could further improve performance, such as closed-loop fine-tuning and top-$k$ sampling, remain unexplored.
- Discretization of the action space may limit trajectory precision.

## Related Work & Insights

- GATr (NeurIPS'23) introduced the E(3)-equivariant geometric algebra Transformer; this work efficiently adapts it to 2D.
- SMART achieves equivariance via RPE and holds the top position on the WOSAC leaderboard, but incurs high computational cost.
- DRoPE extends RoPE to 2D but achieves only translation equivariance without rotation equivariance.
- VN-Transformer achieves SO(3)-equivariance via Vector Neurons but must sacrifice strict equivariance for numerical stability.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Novel combination of 2D geometric algebra encoding and equivariant Transformer)
- Experimental Thoroughness: ⭐⭐⭐⭐ (WOSAC benchmark evaluation, scalability analysis, and ablation studies are comprehensive)
- Writing Quality: ⭐⭐⭐⭐⭐ (Mathematical derivations are clear; architectural descriptions are thorough)
- Value: ⭐⭐⭐⭐⭐ (Addresses the efficiency bottleneck of equivariant agent modeling with strong application prospects)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[AAAI 2026\] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction](../../AAAI2026/autonomous_driving/catformer_causal_temporal_transformer_with_dynamic_contextual_fusion_for_driving.md)
- [\[AAAI 2026\] Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models](../../AAAI2026/autonomous_driving/unlocking_efficient_vehicle_dynamics_modeling_via_analytic_world_models.md)
- [\[CVPR 2026\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](colc_communication-efficient_collaborative_perception_with_lidar_completion.md)

</div>

<!-- RELATED:END -->
