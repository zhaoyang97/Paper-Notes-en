---
title: >-
  [Paper Note] SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model
description: >-
  [CVPR 2026][Autonomous Driving][4D occupancy prediction] This paper proposes SparseWorld-TC, a pure attention-based sparse occupancy world model that bypasses VAE discretization and BEV intermediate representations…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "4D occupancy prediction"
  - "world model"
  - "sparse representation"
  - "trajectory conditioning"
  - "pure attention architecture"
date: 2026-05-08
content_hash: 9e88554a3b8b6c08
---

# SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model

**Conference**: CVPR 2026
**arXiv**: [2511.22039](https://arxiv.org/abs/2511.22039)  
**Code**: [GitHub](https://github.com/MrPicklesGG/SparseWorld)  
**Area**: Autonomous Driving / World Models
**Keywords**: 4D occupancy prediction, world model, sparse representation, trajectory conditioning, pure attention architecture

## TL;DR

This paper proposes SparseWorld-TC, a pure attention-based sparse occupancy world model that bypasses VAE discretization and BEV intermediate representations, directly predicting trajectory-conditioned multi-frame future occupancy end-to-end from raw image features, achieving substantial improvements over existing methods on nuScenes.

## Background & Motivation

Occupancy world models predict future 3D scene occupancy to understand environmental dynamics, playing a critical role in autonomous driving. Existing approaches suffer from two primary limitations:

1. **VAE discretization bottleneck**: Methods such as OccWorld and OccLLaMA employ VQ-VAE to encode continuous 3D scene data into discrete tokens with a fixed codebook, limiting representational capacity and discarding fine-grained information.
2. **BEV intermediate representation constraints**: Most methods rely on dense BEV feature maps for spatiotemporal modeling, imposing explicit geometric constraints that restrict flexible interaction across different scales.

Motivated by the success of pure attention architectures (e.g., GPT, VGGT) in language and 3D visual domains, the authors ask: can a fully attention-based feedforward architecture capture spatiotemporal dependencies directly from raw image features via sparse occupancy representations?

## Method

### Overall Architecture

Historical multi-frame images → image backbone feature extraction → deformable attention sampling of sensor embeddings → alternating frame-level attention and temporal attention to fuse occupancy / sensor / trajectory embeddings → MLP decoding of per-anchor offsets and semantic labels → output multi-frame future occupancy predictions.

### Key Designs

1. **Sparse Occupancy Representation**:
    - Function: Represents scene occupancy as a set of anchors, each containing a group of randomly initialized 3D points and an associated feature vector.
    - Mechanism: Each anchor feature vector is decoded by an MLP to predict per-point 3D offsets and semantic labels, effectively "denoising" random points into a coherent occupancy field.
    - Design Motivation: Avoids the fixed-resolution limitation of BEV and the discretization information loss of VAE, maintaining full sparsity and flexibility.

2. **Trajectory Spatiotemporal Embedding**:
    - Function: Encodes trajectory waypoints as feature vectors serving as conditioning signals.
    - Mechanism: Combines positional embeddings (MLP projection of 16-dimensional homogeneous matrices) and temporal embeddings (sinusoidal positional encoding), fused via affine transformation.
    - Design Motivation: Inspired by MLN, enables the model to condition on arbitrary future trajectories and support waypoints at varying time intervals.

3. **Pure Attention Fusion Architecture**:
    - Function: Uniformly fuses occupancy, sensor, and trajectory embeddings.
    - Mechanism: Stacks frame-level attention modules (cross-attention between occupancy and sensor embeddings, plus trajectory self-attention) and temporal attention modules (cross-frame self-attention), iteratively refined over multiple passes.
    - Design Motivation: Once all modalities are projected into a unified embedding space, standard attention mechanisms effectively capture long-range spatiotemporal dependencies.

### Loss & Training

- Chamfer Distance loss supervises alignment between predicted points and GT occupancy voxel centers.
- Focal classification loss supervises semantic predictions.
- **Random set strategy**: During training, the number of predicted frames $L \in \{2, \ldots, T\}$ is sampled randomly, enabling the model to generalize across different prediction horizon requirements.

## Key Experimental Results

### Main Results (Occ3D-nuScenes, Camera Input)

| Method | 1s mIoU | 2s mIoU | 3s mIoU | Avg. mIoU | Avg. IoU |
|--------|---------|---------|---------|-----------|----------|
| COME | 26.56 | 21.73 | 18.49 | 22.26 | 44.07 |
| Ours-Small | 27.95 | 25.51 | 23.35 | 25.60 | 49.02 |
| Ours-Large | 28.64 | 26.28 | 24.36 | 26.42 | 49.21 |
| Ours-Large* (DINOv3) | 32.76 | 29.62 | 27.28 | 29.89 | 53.52 |

### Long-Term Prediction (8 seconds)

| Method | Input | Avg. mIoU | Avg. IoU |
|--------|-------|-----------|----------|
| COME | Occ GT | 19.07 | 29.96 |
| Ours-Large | Camera | 22.33 | 45.35 |

### Ablation Study

| Configuration | Avg. mIoU | Avg. IoU | Note |
|---------------|-----------|----------|------|
| No trajectory | 15.44 | 32.19 | Trajectory conditioning is critical |
| Predicted trajectory | 21.57 | 44.76 | Predicted trajectory remains effective |
| GT trajectory | 25.60 | 49.02 | More accurate trajectory yields consistent gains |
| Fixed-frame training | 20.36 | 43.25 | Random set strategy is superior |

### Key Findings

- Using only camera input, the model surpasses DOME which relies on GT occupancy input (mIoU 29.89 vs. 27.10).
- Performance degradation over long-term prediction is substantially smaller than existing methods, with IoU still reaching 39.97 at 8 seconds.
- The Small variant runs 2.6× faster than the Large variant with marginal performance difference, enabling a favorable efficiency–accuracy trade-off.

## Highlights & Insights

- The first pure attention-based occupancy world model to fully bypass both VAE and BEV, offering a clean and principled design paradigm.
- The flexibility of sparse representation allows the model to scale to varying anchor counts and extended prediction horizons.
- Significant advantage in long-term prediction: performance barely degrades beyond 3 seconds, whereas existing methods decline sharply.
- The framework directly benefits from large-scale foundation models such as DINOv3 for further performance gains.

## Limitations & Future Work

- Sparse representation may be less effective than dense methods at recovering extremely fine-grained scene details.
- Computational cost grows with the number of anchors; the Large variant achieves only 3.58 FPS.
- Single-GT evaluation for long-term prediction is inherently limited due to the multi-modal nature of future scenes.
- Joint training with downstream planning modules remains unexplored.

## Related Work & Insights

- **vs. OccWorld / OccLLaMA**: These methods rely on VAE discretization and autoregressive generation, constrained by codebook capacity; the proposed method is end-to-end without discretization.
- **vs. DOME / COME**: These methods use diffusion models with BEV and continuous VAE; the proposed method performs feedforward single-pass inference, which is more efficient.
- **vs. VGGT**: Borrows the pure attention architecture philosophy but is specifically designed for 4D occupancy prediction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First pure attention-based sparse occupancy world model; introduces an entirely new design paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers short- and long-term prediction, ablations, and visualizations with comprehensive baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear architecture presentation, concise formulations, and well-motivated design choices.
- Value: ⭐⭐⭐⭐⭐ — Establishes a novel sparse attention paradigm for occupancy world models with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)
- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](../../ICCV2025/autonomous_driving/langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[ICLR 2026\] Astra: General Interactive World Model with Autoregressive Denoising](../../ICLR2026/autonomous_driving/astra_general_interactive_world_model_with_autoregressive_denoising.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
