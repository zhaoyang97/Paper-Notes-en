---
title: >-
  [Paper Note] Seeing the Wind from a Falling Leaf
description: >-
  [NeurIPS 2025][Video Generation][Invisible Force Field Recovery] An end-to-end differentiable inverse graphics framework is proposed that jointly models object geometry/physical properties, force field representations, and physical processes to recover invisible force fields (e.g., wind fields) from video via backpropagation, while supporting physics-based video generation and editing.
tags:
  - NeurIPS 2025
  - Video Generation
  - Invisible Force Field Recovery
  - Differentiable Physics Simulation
  - Inverse Graphics
  - 3D Gaussian
  - Causal Triplane
date: 2026-05-08
content_hash: 01ad70864b062f35
---

# Seeing the Wind from a Falling Leaf

**Conference**: NeurIPS 2025
**arXiv**: [2512.00762](https://arxiv.org/abs/2512.00762)
**Code**: [Project Page](https://chaoren2357.github.io/seeingthewind/)
**Area**: Video Generation
**Keywords**: Invisible Force Field Recovery, Differentiable Physics Simulation, Inverse Graphics, 3D Gaussian, Causal Triplane

## TL;DR

An end-to-end differentiable inverse graphics framework is proposed that jointly models object geometry/physical properties, force field representations, and physical processes to recover invisible force fields (e.g., wind fields) from video via backpropagation, while supporting physics-based video generation and editing.

## Background & Motivation

1. **Background**: Computer vision has long pursued motion modeling from video, yet the invisible physical interactions (forces) underlying motion remain largely unexplored. System identification methods only estimate a small number of physical parameters (e.g., mass, friction coefficients).

2. **Limitations of Prior Work**: Force estimation is far more challenging than physical parameter estimation — forces are full vectors that can exist throughout 3D space in dense and complex configurations. Time integration in physical simulators leads to gradient explosion/vanishing, making backpropagation unstable.

3. **Key Challenge**: Inferring invisible forces from visible motion is an ill-posed inverse problem with incomplete information. Conventional photometric loss provides insufficient gradients, and dense 3D scene flow is highly noisy.

4. **Goal**: Recover force fields driving object motion from video input alone, without manually specifying forces or environmental conditions.

5. **Key Insight**: Construct a fully differentiable "perception → physics → optimization" pipeline, replacing dense scene flow with sparse keypoint tracking to substantially reduce the optimization space dimensionality and stabilize gradients.

6. **Core Idea**: 3D Gaussians (Lagrangian particles) + Causal Triplane (Eulerian force field) + MPM simulator + sparse tracking objective = an end-to-end differentiable pipeline from video to force fields.

## Method

### Overall Architecture

Four modules: (1) Object modeling (3D Gaussians + VLM physical properties) → (2) Force field representation (Causal Triplane) → (3) Physical process (differentiable MPM simulator) → (4) Sparse tracking optimization (backpropagation to recover force fields).

### Key Designs

**1. 3D Gaussian-based Object Modeling + VLM Physical Properties**

- **Function**: Unified representation of object geometry, appearance, and physical properties.
- **Mechanism**: Each 3D Gaussian $G = \{\mathbf{x}, \mathbf{v}, \Sigma, \sigma, SH, \mathbf{D}, m, E, \nu\}$ encodes position, velocity, shape, appearance, and physical attributes (mass $m$, Young's modulus $E$, Poisson's ratio $\nu$). Gaussians are initialized from the first frame only (via a metric depth model + Gaussian splatting). Physical properties are inferred by GPT-4V from object type recognized in the image and assigned commonsense values; Grounded SAM is used to segment each Gaussian's object assignment.
- **Design Motivation**: 3D Gaussians as Lagrangian particles are naturally compatible with MPM, and VLM commonsense physical knowledge is sufficiently robust for common object types.

**2. Causal Triplane Force Field Representation**

- **Function**: High-fidelity modeling of spatiotemporal continuity and causal dependencies of force fields.
- **Mechanism**: Force is defined as $\mathbf{f}(\mathbf{x}, t) = \mathcal{D}(\gamma(\mathbf{x}) + \varphi(t; \varphi(t-1)))$, where $\gamma(\cdot)$ denotes triplane spatial features, $\varphi(\cdot)$ is a recurrent temporal encoder (initializing the current-timestep MLP with the previous timestep's weights), and $\mathcal{D}$ is a feature decoder. The recurrent dependency of the temporal encoder achieves causal evolution of forces.
- **Design Motivation**: Compared to other 4D representations (K-Planes, HexPlane), the causal triplane decouples space and time, is computationally efficient, and naturally models the temporal causality of forces.

**3. 4D Sparse Tracking Objective**

- **Function**: Stabilize differentiable physics optimization and reduce the prediction space dimensionality.
- **Mechanism**: CoTracker is used to obtain sparse pixel keypoint motions $\mathbf{p}^t \to \mathbf{p}^{t+1}$, which are back-projected to 3D as $\mathbf{P}^t$. Robust 3D keypoint motions $\mathbf{P}^t \to \mathbf{P}^{t+1}$ are obtained by minimizing reprojection error with an ARAP constraint ($\mathcal{L}_{arap}$). Keypoints control all Gaussian motions via barycentric interpolation: $\hat{\mathbf{x}} = \alpha_i \mathbf{P}_i + \alpha_j \mathbf{P}_j + \alpha_k \mathbf{P}_k$.
- **Design Motivation**: Photometric loss suffers from gradient vanishing, and dense 3D scene flow is highly noisy. Sparse keypoints substantially reduce the prediction space (from $N$ Gaussians to $N_{key}$ keypoints), and pixel-level tracking by CoTracker is more reliable than inter-frame depth estimation.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{motion} + \lambda_1 \mathcal{L}_{space} + \lambda_2 \mathcal{L}_{time}$$

- $\mathcal{L}_{motion} = |\hat{\mathbf{x}}^{t+1} - \mathbf{x}^{t+1}|$: tracking motion matching
- $\mathcal{L}_{space}$: spatial total variation regularization
- $\mathcal{L}_{time} = |\varphi_\theta^{t+1} - \varphi_\theta^t|$: temporal smoothness regularization

## Key Experimental Results

### Main Results

**Force Recovery on Synthetic Scenes**

| Material Type | Object | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Magnitude Error (%) ↓ | Direction Error (°) ↓ |
|--------------|--------|--------|--------|---------|----------------------|----------------------|
| Elastic | Lego | 33.70 | 0.98 | 0.01 | 19.53 | 7.02 |
| Elastic | Ficus | 25.92 | 0.94 | 0.03 | 23.97 | 11.55 |
| Elastic | Sunflower | 34.08 | 0.99 | 0.01 | 14.38 | 7.85 |
| Elastoplastic | Toy | 41.35 | 0.99 | 0.00 | 29.19 | 8.11 |
| Elastoplastic | Chair | 40.10 | 0.99 | 0.00 | 33.31 | 23.40 |
| Viscoplastic | Hotdog | 30.63 | 0.96 | 0.02 | 15.09 | 11.63 |

### Ablation Study

| Method | PSNR ↑ | Magnitude Error (%) ↓ | Direction Error (°) ↓ |
|--------|--------|----------------------|----------------------|
| Point force representation | 20.57 | 95.91 | 76.48 |
| Dense scene flow objective | — | Poor | Poor |
| Photometric objective | — | Gradient vanishing | Failed |
| **Sparse tracking + Causal Triplane** | **33.70** | **19.53** | **7.02** |

### Key Findings

- Elastic materials yield the best force recovery (magnitude error 14–24%, direction error 7–12°).
- Elastoplastic materials exhibit larger direction errors (23.4° on Chair) due to additional uncertainty introduced by plastic deformation.
- VLM-estimated physical properties are sufficiently robust for force recovery — even imprecise estimates yield reasonable force fields.
- Force visualizations on real-world videos are physically plausible, and recovered force fields can be applied to new objects for physics-based simulation.

## Highlights & Insights

- This work is the first to recover distributed force fields from video (rather than contact forces or a small number of parameters), making the problem formulation itself a significant contribution.
- The pairing of 3D Gaussians (Lagrangian) and triplane (Eulerian) representations perfectly matches the MPM formalism.
- The sparse tracking objective is the key enabler of practical differentiable physics optimization — it simultaneously reduces optimization dimensionality and improves tracking robustness.
- The application scenario is novel: applying recovered force fields to new objects enables physics-driven video editing.

## Limitations & Future Work

- The sparse tracking objective is primarily suited to objects undergoing bending deformation or small deformations.
- 3D Gaussians initialized from a single frame are incomplete (occluded surface information is missing).
- Physical property estimation relies on VLM commonsense knowledge, which may be inaccurate for unconventional objects.
- Multi-object collision interactions are not currently handled.
- Computational cost is high due to per-frame optimization of force field parameters.

## Related Work & Insights

- This work extends the tradition of differentiable physics inverse problems (e.g., GradSim, PAC-NeRF), advancing from parameter estimation to force field recovery.
- Methods such as PhysDreamer and Physics3D use generative models to drive physical animation but require manually specified forces — the proposed method recovers forces automatically.
- Insight: Combining the rendering capability of 3DGS with the physical simulation capability of MPM may catalyze a new paradigm of physics-aware video generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The problem formulation of recovering force fields from video is highly pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic and real-world scenes with diverse materials and complete ablations, though quantitative evaluation is limited to synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The opening with a Rossetti verse is elegant, and the technical exposition is clear.
- **Value**: ⭐⭐⭐⭐⭐ Bridges vision and physics, opening a new research direction from perception to force field recovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Seeing the Unseen: Zooming in the Dark with Event Cameras](../../AAAI2026/video_generation/seeing_the_unseen_zooming_in_the_dark_with_event_cameras.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](../../CVPR2026/video_generation/seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[NeurIPS 2025\] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals](force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)
- [\[NeurIPS 2025\] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation](lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)

</div>

<!-- RELATED:END -->
