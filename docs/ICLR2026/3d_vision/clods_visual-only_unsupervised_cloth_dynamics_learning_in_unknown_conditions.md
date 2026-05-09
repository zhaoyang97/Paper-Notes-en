---
title: >-
  [Paper Note] CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions
description: >-
  [ICLR 2026][3D Vision][cloth dynamics] CloDS proposes the first framework for unsupervised cloth dynamics learning from multi-view videos. By introducing Spatial Mapping Gaussian Splatting (SMGS) to establish a differentiable mapping between 2D images and 3D meshes, combined with dual-position opacity modulation to address self-occlusion, the method enables a GNN to learn cloth dynamics approaching fully supervised performance without any physical parameter supervision.
tags:
  - ICLR 2026
  - 3D Vision
  - cloth dynamics
  - unsupervised learning
  - Gaussian splatting
  - differentiable rendering
  - intuitive physics
date: 2026-05-08
content_hash: 77cb04ef463df4e5
---

# CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions

**Conference**: ICLR 2026  
**arXiv**: [2602.01844](https://arxiv.org/abs/2602.01844)  
**Code**: [https://github.com/whynot-zyl/CloDS](https://github.com/whynot-zyl/CloDS)  
**Area**: 3D Vision  
**Keywords**: cloth dynamics, unsupervised learning, Gaussian splatting, differentiable rendering, intuitive physics

## TL;DR
CloDS proposes the first framework for unsupervised cloth dynamics learning from multi-view videos. By introducing Spatial Mapping Gaussian Splatting (SMGS) to establish a differentiable mapping between 2D images and 3D meshes, combined with dual-position opacity modulation to address self-occlusion, the method enables a GNN to learn cloth dynamics approaching fully supervised performance without any physical parameter supervision.

## Background & Motivation

**State of the Field**: Deep learning has achieved significant progress in simulating dynamic systems (fluids, cloth, multi-body dynamics), but existing methods heavily rely on known physical properties as supervision signals (e.g., particle positions, mesh node coordinates).

**Limitations of Prior Work**:
   - In real-world scenarios, physical properties (material parameters, environmental conditions) are often unknown, limiting practical applicability.
   - Intuitive physics methods (learning dynamics from visual observations) primarily target rigid body interactions and perform poorly on continuum mechanics, especially cloth.
   - Dynamic scene novel view synthesis methods fail to generalize to unseen frames; video prediction methods struggle to maintain temporal consistency under frequent self-occlusion.

**Root Cause**: Cloth has an infinite-dimensional state space, complex self-occlusion, and large nonlinear deformations. Existing particle representations (e.g., NeuroFluid) are ill-suited for the thin-sheet structure of cloth, while directly applying mesh-based Gaussian Splatting introduces perspective distortion due to self-occlusion.

**Paper Goals**:
   - Define and solve the Cloth Dynamics Grounding (CDG) problem: unsupervised cloth dynamics learning from multi-view videos.
   - Design a differentiable 2D↔3D mapping that enables GNN dynamics models to be trained with pixel-level losses.
   - Resolve rendering distortions under large deformations and severe self-occlusion.

**Starting Point**: Decompose the problem into the joint learning of three probabilistic models — rendering $p(Y_t|M_t)$, inverse rendering $p(M_t|Y_{1:t})$, and dynamics transition $p(M_{t+1}|M_t)$ — chained through a Differentiable Visual Computing (DVC) framework.

**Core Idea**: Establish a differentiable, temporally consistent 2D-3D mapping via mesh-based Gaussian Splatting with dual-position opacity modulation, inverting 3D mesh sequences from video as pseudo-labels for dynamics learning.

## Method

### Overall Architecture
A three-stage pipeline: (1) construct Gaussian components from the first-frame mesh and multi-view images; (2) recover 3D mesh sequences $\tilde{M}_{1:T}$ from subsequent frames via backpropagation; (3) train a GNN dynamics model on the recovered mesh sequences.

### Key Designs

1. **Spatial Mapping Gaussian Splatting (SMGS)**:

    - **Function**: Establish a temporally consistent differentiable 2D-3D mapping.
    - **Mechanism**: Anchor Gaussian components to mesh faces, with each Gaussian center determined by barycentric interpolation $\mu_t = \beta_1 X_{t,1}^W + \beta_2 X_{t,2}^W + \beta_3 X_{t,3}^W$. As the mesh deforms, Gaussians update automatically using the same $\beta$ coefficients and new vertex positions. Rotation matrices are determined by face normals and Gram-Schmidt orthogonalization; scales are determined by edge lengths.
    - **Design Motivation**: Based on GaMeS, but GaMeS cannot handle self-occlusion under large deformations. The key innovation of SMGS is dual-position opacity modulation.

2. **Dual-Position Opacity Modulation**:

    - **Function**: Dynamically control the opacity of each Gaussian component via $\alpha_{i,t} = f_\theta(\mu_{i,t}^W, \mu_{i,t}^M)$.
    - **Mechanism**: Opacity is modulated by jointly considering world coordinates (relative position $\mu^W$) and mesh coordinates (absolute position $\mu^M$), where $f_\theta$ is an MLP. World coordinates prevent perspective distortion (correctly assigning weights under overlap); mesh coordinates prevent cloth from becoming transparent when it moves to previously unobserved regions.
    - **Design Motivation**: GaMeS does not adjust opacity during motion, leading to incorrect weights in occluded regions and rendering errors. Using world coordinates alone causes transparency in novel regions; combining both addresses both failure modes simultaneously.

3. **Inverse Rendering for 3D Label Recovery**:

    - **Function**: Recover 3D mesh node positions from 2D images via backpropagation.
    - **Mechanism**: Given the current mesh $M_t$, optimize displacement $\Delta x_t^W$ such that the rendered result $\tilde{I}_{t+1}$ matches the target image $Y_{t+1}$. Applied recursively, the full sequence $\tilde{M}_{1:T}$ can be recovered from the first-frame mesh.
    - **Design Motivation**: No 3D mesh annotations (e.g., physics simulator outputs) are required; multi-view video alone suffices to obtain pseudo-labels for training the dynamics model.

### Loss & Training
- **Stage 1** (Gaussian construction): Standard 3DGS loss $\mathcal{L}_{render} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$, $\lambda=0.2$.
- **Stage 2** (mesh recovery): $\mathcal{L}_{geometry} = \mathcal{L}_1(\text{SMGS}(\tilde{x}_{t+1}^W), Y_{t+1}) + \gamma\mathcal{L}_{edge}$, where the edge loss preserves inter-node distances to prevent excessive deformation.
- **Stage 3** (dynamics training): $\mathcal{L}_{node} = \sum_{t=1}^T \text{MSE}(\hat{x}_t^W, x_t^W)$, rollout length $T=8$.

## Key Experimental Results

### Main Results (Cloth Dynamics Learning — RMSE)

| Method | Supervision | Viewed Interp. | Viewed Extrap. | Unviewed Interp. | Unviewed Extrap. |
|--------|-------------|----------------|----------------|------------------|------------------|
| MGN | All mesh | 0.1286 | 0.1291 | 0.1358 | 0.1314 |
| MGN* | 50 meshes | 0.1380 | 0.1388 | 0.1460 | 0.1362 |
| CloDS | 50 mesh + 50 video | 0.1321 | 0.1344 | 0.1399 | 0.1339 |
| CloDS** | All video | 0.1294 | 0.1307 | 0.1388 | 0.1325 |

### Dynamic Scene Novel View Synthesis

| Model | PSNR↑ | SSIM×10↑ | LPIPS×1000↓ |
|-------|-------|----------|-------------|
| 4DGS | 23.21 | 9.718 | 15.82 |
| GaMeS | 33.02 | 9.937 | 5.21 |
| **SMGS (Ours)** | **36.24** | **9.959** | **3.53** |
| 3DGS (upper bound) | 39.63 | 9.986 | 2.53 |

### Key Findings
- **CloDS learns dynamics from video approaching fully supervised performance**: CloDS** trained on all videos achieves RMSE very close to MGN trained on all meshes (gap <5%), demonstrating the viability of the unsupervised approach.
- **SMGS substantially outperforms baseline rendering methods**: PSNR exceeds GaMeS by 3.2 dB and 4DGS by 13 dB.
- **Both components of dual-position opacity modulation are indispensable**: removing world coordinates causes perspective distortion; removing mesh coordinates causes transparency in moving regions.
- **Strong generalization**: On cloth with unseen initial states, CloDS extrapolation RMSE is only ~1% higher than on viewed configurations.

## Highlights & Insights
- **Introducing the DVC framework for cloth dynamics is a far-sighted contribution**: It establishes a complete closed loop of rendering → inverse rendering → dynamics, enabling physical dynamics to be learned from only the first-frame mesh and multi-view video. This paradigm is transferable to any scenario requiring physical model learning from visual observations.
- **Dual-position opacity modulation is simple yet critical**: By feeding both world and mesh coordinates into a single MLP to modulate opacity, the method resolves rendering distortion under self-occlusion with minimal computational overhead.
- **The decoupled three-stage training design is highly practical**: Stages 1 and 2 require no temporal loss (single-step training); only Stage 3 uses rollout. This allows the first two stages to be parallelized, substantially reducing training complexity.

## Limitations & Future Work
- Validation is conducted solely on Blender synthetic datasets; noise, lighting variation, and multi-object interaction in real-world scenes remain insufficiently tested.
- Initial mesh estimation requires a first-frame mesh as prior, which may be unavailable in real-world settings.
- The GNN dynamics model (MGN) is fixed; stronger dynamics models (e.g., Transformer-based) are not explored.
- Only single-piece cloth is handled; cloth–object interactions (e.g., dressing) are not considered.
- Cloth trajectories in the dataset are relatively simple (flag); more complex scenarios (e.g., garment folding) may require substantially more training data.

## Related Work & Insights
- **vs. NeuroFluid**: NeuroFluid learns fluid dynamics via particle representations and differentiable rendering, but particles are ill-suited for the thin-sheet structure of cloth; CloDS adopts a mesh representation more appropriate for cloth.
- **vs. GaMeS**: Both use mesh-based Gaussian Splatting, but GaMeS does not handle opacity changes during motion, leading to incorrect rendering in occluded regions.
- **vs. 4DGS/M5D-GS**: 4D dynamic scene methods perform poorly under large deformations and do not learn generalizable dynamics models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneers the CDG problem and a corresponding solution framework; dual-position opacity modulation is a notable innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple tasks, but validation is limited to synthetic data.
- Writing Quality: ⭐⭐⭐⭐ Mathematical modeling is clear, though notation is somewhat dense.
- Value: ⭐⭐⭐⭐ Opens a new direction for learning cloth dynamics from visual observations, with potential transfer to robotic manipulation and related applications.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image](../../CVPR2026/3d_vision/zero-shot_reconstruction_of_animatable_3d_avatars_with_cloth_dynamics_from_a_sin.md)
- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)
- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)
- [\[CVPR 2026\] OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](../../CVPR2026/3d_vision/openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)
- [\[ICLR 2026\] Quantized Visual Geometry Grounded Transformer](quantized_visual_geometry_grounded_transformer.md)

<!-- RELATED:END -->
