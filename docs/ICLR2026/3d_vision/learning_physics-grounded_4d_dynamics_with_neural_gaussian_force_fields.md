---
title: >-
  [Paper Note] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields
description: >-
  [ICLR 2026][3D Vision][3D Gaussian Splatting] This paper proposes the NGFF framework, which reconstructs 3D Gaussian representations from multi-view RGB images and learns explicit neural force fields to drive physics-based dynamics. By solving ODEs, the framework enables interactive, physically plausible 4D video generation that is two orders of magnitude faster than traditional Gaussian simulators, surpassing Veo3 and NVIDIA Cosmos.
tags:
  - ICLR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - Force Field Learning
  - Physical Reasoning
  - 4D Video Prediction
  - Neural Operator
date: 2026-05-08
content_hash: 97f07a10778ed9a5
---

# Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields

**Conference**: ICLR 2026
**arXiv**: [2602.00148](https://arxiv.org/abs/2602.00148)
**Code**: [Project Page](https://neuralgaussianforcefield.github.io/)
**Area**: 3D Vision / Physics Simulation
**Keywords**: 3D Gaussian Splatting, Force Field Learning, Physical Reasoning, 4D Video Prediction, Neural Operator

## TL;DR
This paper proposes the NGFF framework, which reconstructs 3D Gaussian representations from multi-view RGB images and learns explicit neural force fields to drive physics-based dynamics. By solving ODEs, the framework enables interactive, physically plausible 4D video generation that is two orders of magnitude faster than traditional Gaussian simulators, surpassing Veo3 and NVIDIA Cosmos.

## Background & Motivation

**Background**: Video generation models produce visually stunning outputs but lack physical understanding, frequently violating fundamental laws such as gravity and object permanence. Methods combining 3DGS with traditional physics engines achieve good physical consistency but at prohibitive computational cost.

**Limitations of Prior Work**: (1) Particle/mesh-based methods require predefined physical models and structured inputs, resulting in poor generalization; (2) MPM-based Gaussian methods offer high physical fidelity but at unacceptable computational cost; (3) Large video models overfit to surface visual features rather than learning physical principles.

**Key Challenge**: There is a need for an approach that simultaneously achieves physical consistency (via force modeling), computational efficiency (without MPM), and the ability to learn directly from visual observations (without structured inputs).

**Key Insight**: Rather than predefining physical models, the paper learns explicit force fields — using a neural operator to predict inter-object forces and integrating dynamics via ODEs. 3D Gaussians provide an object-aware representational interface.

## Method

### Overall Architecture
Multi-view RGB → feed-forward 3D Gaussian reconstruction (SAM2 segmentation + DiffSplat refinement) → PointNet object feature encoding → DeepONet force field prediction → ODE integration for dynamics simulation → Gaussian rendering for video generation.

### Key Designs

1. **Object-Aware 3D Reconstruction**:

    - Function: A feed-forward Transformer constructs 3D Gaussians from multi-view RGB inputs, with SAM2 segmenting them into individual objects.
    - Mechanism: DINOv2 features → alternating-attention Transformer → prediction of camera poses and Gaussian parameters. DiffSplat completes occluded regions.
    - Design Motivation: Physics simulation requires object-level decomposed representations.

2. **Neural Gaussian Force Field (NGFF)**:

    - Function: Uses a neural operator to predict global transformation forces and local stress fields between objects.
    - Mechanism: Global force $\mathbf{F}^{\text{global}}(\mathbf{z}^q(t)) = \sum_{i \in \mathcal{N}(q)} \mathbf{W}(f_\eta(\mathbf{z}^i) \odot f_\phi(\mathbf{z}^q)) + \mathbf{b}$; local stress $\mathbf{F}^{\text{local}} = \Phi(\mathbf{F}^{\text{latent}}, \text{CAM}, \mathbf{x}^q, \dot{\mathbf{x}}^q)$, where CAM denotes the contact area mask.
    - Design Motivation: Global forces handle rigid-body translation/rotation, while local forces handle soft-body deformation. A relational graph encodes inter-object contact.

3. **ODE Integration Trajectory Decoder**:

    - Function: Integrates object trajectories from the force field using a second-order ODE solver.
    - Mechanism: $\mathbf{z}^q(t) = \text{ODESolve}(\mathbf{z}^q(0), \mathbf{F}, 0, t)$, $\dot{\mathbf{s}}(t) = \dot{\mathbf{s}}(0) + \int_0^t \mathbf{F}(\mathbf{z}^q(t)) dt$
    - Design Motivation: Provides a fully differentiable bridge connecting force field prediction and dynamics simulation.

### Loss & Training
- Two-stage training: (1) feed-forward reconstruction fine-tuned on WildRGBD; (2) dynamics prediction trained on synthetic MPM data.
- Dynamics loss: MSE between predicted and ground-truth Gaussian configurations and motion trajectories.

## Key Experimental Results

### GSCollision Dataset
- 640K rendered physics videos (~4TB), covering 10 categories of everyday objects (both rigid and soft), encompassing falling, collision, rotation, sliding, and container interaction scenarios.

### Main Results (Dynamics Prediction)

| Model | Spatial RMSE↓ | Temporal RMSE↓ | Combined RMSE↓ | Inference Time↓ |
|-------|--------------|----------------|----------------|-----------------|
| VLM-MPM | High | High | High | >100s |
| Pointformer | Medium | Medium | Medium | Medium |
| **NGFF** | **Lowest** | **Lowest** | **Lowest** | **~1s** |

### Video Generation Comparison

| Model | Physical Consistency | Visual Quality |
|-------|---------------------|----------------|
| Veo3 | Poor (violates physics) | High |
| NVIDIA Cosmos | Poor | High |
| **NGFF** | **Strong** | **Reasonable** |

### Key Findings
- NGFF is two orders of magnitude faster than MPM-based Gaussian simulators, as it learns a force field rather than simulating per-particle dynamics.
- The model generalizes well compositionally (4–6 objects at test time vs. 3 during training) and temporally (beyond training sequence length).
- Explicit force field modeling enables interactive generation — external forces can be applied to alter trajectories.
- Sim-to-real transfer is achieved through the decoupled interface provided by the 3D Gaussian representation.

## Highlights & Insights
- **Learning force fields rather than dynamics**: Instead of directly predicting the next-frame state, the model predicts forces and integrates via ODE to obtain states. This yields better generalization, as force laws are more universal than state transition patterns.
- **Two orders of magnitude speedup**: The key lies in using a neural operator to predict forces in a single forward pass, as opposed to MPM's iterative per-particle simulation.
- **Decoupled global and local force fields**: Rigid-body motion is handled by global forces (translation + rotation), while soft-body deformation is handled by local stress. This physics-motivated decomposition generalizes better than end-to-end learning.

## Limitations & Future Work
- Training data are derived from synthetic MPM simulations, limiting the diversity of real-world physical parameters (e.g., friction, elasticity).
- SAM2 segmentation may fail in scenes with complex occlusions.
- The current framework models only rigid bodies and simple soft bodies; complex materials such as fluids and cloth are not addressed.
- The quality of DiffSplat completion affects downstream dynamics prediction.

## Related Work & Insights
- **vs. Veo3/Cosmos**: Large video models achieve high visual quality but lack physical understanding; NGFF exhibits the opposite trade-off — strong physical consistency with reasonable visual quality.
- **vs. MPM-Gaussian**: MPM is physically accurate but two orders of magnitude slower; NGFF replaces explicit simulation with a neural force field.
- **vs. GNN-based methods**: GNNs learn relational dynamics via graph neural networks, whereas NGFF uses neural operators to learn force fields, yielding clearer physical interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of neural force fields and 3D Gaussians is original with clear physical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐ The GSCollision dataset is comprehensive, with multi-dimensional generalization evaluation.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly presented with intuitive capability demonstrations.
- Value: ⭐⭐⭐⭐⭐ Represents an important step toward physics-driven world models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics](diffwind_physics-informed_differentiable_modeling_of_wind-driven_object_dynamics.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](../../NeurIPS2025/3d_vision/mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[ICLR 2026\] Einstein Fields: A Neural Perspective To Computational General Relativity](einstein_fields_a_neural_perspective_to_computational_general_relativity.md)
- [\[CVPR 2026\] Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields](../../CVPR2026/3d_vision/dynamic_black-hole_emission_tomography_with_physics-informed_neural_fields.md)
- [\[ICLR 2026\] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)

<!-- RELATED:END -->
