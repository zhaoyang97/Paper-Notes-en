---
title: >-
  [Paper Note] Toward Material-Agnostic System Identification from Videos
description: >-
  [ICCV2025][system identification] This paper proposes MASIV, the first visual system identification framework that requires no predefined material priors. It replaces hand-crafted elastic/plastic equations with a learnable neural constitutive model, reconstructs dense continuum particle trajectories to provide temporally rich geometric supervision, and infers the intrinsic dynamic properties of objects from multi-view videos.
tags:
  - "ICCV2025"
  - "system identification"
  - "neural constitutive model"
  - "material point method"
  - "3D Gaussian splatting"
  - "differentiable simulation"
date: 2026-05-08
content_hash: bd147d90ceb9ea81
---

# Toward Material-Agnostic System Identification from Videos

**Conference**: ICCV2025
**arXiv**: [2508.01112](https://arxiv.org/abs/2508.01112)  
**Code**: [Skaldak/MASIV](https://github.com/Skaldak/MASIV)  
**Area**: Physical Simulation / System Identification / Differentiable Rendering
**Keywords**: system identification, neural constitutive model, material point method, 3D Gaussian splatting, differentiable simulation

## TL;DR

This paper proposes MASIV, the first visual system identification framework that requires no predefined material priors. It replaces hand-crafted elastic/plastic equations with a learnable neural constitutive model, reconstructs dense continuum particle trajectories to provide temporally rich geometric supervision, and infers the intrinsic dynamic properties of objects from multi-view videos.

## Background & Motivation

### Visual System Identification

System identification aims to recover object geometry and the physical laws governing its motion from visual observations. Typical approaches integrate differentiable rendering (NeRF, 3DGS) with differentiable simulators (e.g., MPM) end-to-end, fitting parametric physical models to observations via optimization.

### Material Dependency in Prior Methods

Existing methods (PAC-NeRF, Spring-Gaus, NeuMA, GIC) all rely on **material-specific constitutive laws**:
- They require prior knowledge of the material type (elastomer, plasticine, sand, fluid, etc.)
- They employ hand-crafted elastic/plastic models (e.g., neo-Hookean elasticity + identity plastic return mapping)
- They estimate only a small number of physical parameters (Young's modulus, viscosity, friction angle, etc.)

Key limitations:
1. Restricted applicability in scenarios with unknown materials
2. Require selecting an appropriate constitutive model for each material type
3. Cannot generalize to in-the-wild scenes where material properties are unknown

### Core Challenge

Directly applying neural constitutive laws (NCLaw) to visual system identification poses a significant challenge: NCLaw assumes access to complete particle state information (position, velocity, deformation gradient, affine momentum), which cannot be obtained from visual observations alone. Supervision based solely on per-frame pixel losses provides insufficient constraints, leading to unstable optimization and physically implausible behavior.

## Method

### Overall Pipeline (Three Phases)

**Phase I – Geometric Reconstruction**: Reconstruct dynamic Gaussians and dense particle trajectories from multi-view videos.
**Phase II – Material-Agnostic System Identification**: Learn a neural constitutive model using visual observations and reconstructed motion cues.
**Phase III – Generalizable Digital Twin**: Obtain a digital twin capable of simulating novel interactions (new velocities/forces).

### Phase I: Dynamic Gaussian Reconstruction

A set of canonical Gaussian kernels is maintained and warped temporally via a deformation network:

- **Basis network**: maps time steps to $B$ bases, producing deformation bases for position and scale.
- **Coefficient network**: estimates per-basis weights from canonical Gaussian centers and time steps.
- **Optimization objective**: L1 + SSIM loss + scale regularization.

### Phase I (Continued): Continuum Trajectory Estimation

This is one of the key innovations. Gaussian particles are converted into a solid continuum, and temporally dense particle trajectories are estimated:

1. Interior volumes are filled following the GIC approach to form continuum particles with uniform density.
2. The deformation network is fine-tuned to accommodate continuum particles (including interior ones) using Chamfer distance loss.
3. Temporal positional encodings of the motion basis functions enable temporal interpolation.
4. Particle positions are inferred at every simulation time step ($N = T/\tau \gg T$, far exceeding the number of video frames), serving as pseudo ground truth for subsequent optimization.

### Phase II: Material-Agnostic System Identification

#### Neural Constitutive Model

Following NCLaw, two MLPs are used to parameterize the elastic and plastic constitutive laws:

- **Elastic constitutive model**: computes the first Piola–Kirchhoff stress from the elastic deformation gradient.
- **Plastic constitutive model**: enforces plasticity constraints on the trial elastic deformation gradient.
- **Physical priors**: frame invariance (rotation-invariant input representation) + equilibrium at the undeformed state (bias term elimination).
- **Initialization**: NCLaw pretrained weights are used for stable initialization.

#### MPM State Transition

At each time step: the elastic constitutive model computes stress → time integration updates particle states → the plastic constitutive model corrects the deformation gradient.

#### Optimization Objective

The total loss minimizes geometric loss + silhouette loss:

- **Geometric loss (trajectory supervision)**: L1 loss between simulated particle positions and trajectories predicted by the deformation network, computed at every simulation time step (temporally dense).
- **Silhouette loss**: L1 loss between rendered masks and object silhouettes.

### Why Temporally Dense Geometric Constraints Are Necessary

- Sparse per-frame supervision (at the surface or continuum level) is insufficient to constrain complex neural constitutive models.
- Inter-step behavior remains unconstrained, potentially leading to overfitting and physically implausible deformations.
- Trajectory-level dense supervision enforces constraints at every simulation step, yielding the lowest and most stable errors.

## Key Experimental Results

### Experimental Setup

- **Datasets**: PAC-NeRF (synthetic; elasticity, plasticine, sand, Newtonian fluid, non-Newtonian fluid); Spring-Gaus (synthetic + real; elastomers).
- **Baselines**: PAC-NeRF, Spring-Gaus, NeuMA, GIC (all require material priors).
- **Metrics**: Chamfer Distance (CD), PSNR, SSIM.
- **Hardware**: Single NVIDIA A100 GPU.

### Main Results

**Observable-state simulation on PAC-NeRF dataset (CD metric):**

| Method | Newtonian | Non-Newtonian | Elasticity | Plasticine | Sand | Avg. |
|--------|-----------|---------------|------------|------------|------|------|
| PAC-NeRF | 0.277 | 0.236 | 0.238 | 0.429 | 0.212 | 0.278 |
| GIC | 0.243 | 0.195 | 0.178 | 0.196 | 0.250 | 0.212 |
| **MASIV** | **0.233** | **0.198** | **0.192** | **0.201** | **0.229** | **0.210** |

**Spring-Gaus synthetic dataset (CD metric):**

| Method | Avg. CD |
|--------|---------|
| PAC-NeRF | 2.11 |
| Spring-Gaus | 0.85 |
| GIC | 0.17 |
| **MASIV** | **0.13** |

MASIV achieves superior geometric accuracy over all material-prior-dependent baselines without using any material priors.

**Spring-Gaus real-world dataset:**

| Method | Avg. PSNR | Avg. SSIM |
|--------|-----------|-----------|
| Spring-Gaus | 32.90 | 0.994 |
| GIC | 38.11 | 0.996 |
| **MASIV** | **41.12** | **0.997** |

### Ablation Study

**Effect of geometric constraint type:**

| Geometric Constraint | Characteristics | Performance |
|----------------------|-----------------|-------------|
| None | Silhouette supervision only | Highest CD error, unstable |
| Surface | Per-frame surface alignment | Moderate error reduction, unstable |
| Continuum | Per-frame continuum alignment | Further reduction, unstable on some materials |
| **Trajectory** | Per-simulation-step alignment | **Lowest and most stable error** |

Qualitative analysis shows that sparse supervision can lead to physically implausible deformations (e.g., anomalous expansion of butter between simulation steps); trajectory supervision effectively mitigates this via temporally dense constraints.

## Highlights & Insights

- MASIV is the first visual system identification method that requires no material priors.
- The temporally dense trajectory constraint is the key innovation, effectively stabilizing the optimization of the neural constitutive model.
- MASIV achieves state-of-the-art performance on observable-state simulation across diverse material types.
- The resulting digital twin generalizes to novel initial conditions.
- The method still requires NCLaw pretrained weights for initialization; physical priors are implicitly inherited from NCLaw pretraining and are not entirely prior-free.
- Future state prediction slightly underperforms GIC under data-scarce settings, as GIC's known constitutive model provides category-level regularization that MASIV lacks; MASIV's generalization improves with more data.

## Limitations & Future Work

- Initialization still depends on NCLaw pretrained models.
- Future state prediction is marginally weaker than material-prior-based GIC under limited data.
- Trajectory estimation relies on the interpolation quality of the deformation network.
- Per-scene optimization remains time-consuming.
- Future work could explore multi-scene joint training to learn more generalizable neural constitutive models, rather than per-scene optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](../../ACL2025/others/a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[CVPR 2025\] Scene-Agnostic Pose Regression for Visual Localization](../../CVPR2025/others/scene-agnostic_pose_regression_for_visual_localization.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](../../NeurIPS2025/others/recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[ACL 2025\] GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](../../ACL2025/others/genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)

</div>

<!-- RELATED:END -->
