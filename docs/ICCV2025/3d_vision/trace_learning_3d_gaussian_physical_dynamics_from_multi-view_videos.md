---
title: >-
  [Paper Note] TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] TRACE is a framework that treats each 3D Gaussian kernel as a rigid particle and learns an independent translational-rotational dynamical system for it—comprising a complete…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "physical dynamics learning"
  - "future frame extrapolation"
  - "translational-rotational dynamical systems"
  - "dynamic scene reconstruction"
date: 2026-05-08
content_hash: dcfb4d7265566c04
---

# TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos

**Conference**: ICCV 2025
**arXiv**: [2508.09811](https://arxiv.org/abs/2508.09811)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, physical dynamics learning, future frame extrapolation, translational-rotational dynamical systems, dynamic scene reconstruction

## TL;DR

TRACE is a framework that treats each 3D Gaussian kernel as a rigid particle and learns an independent translational-rotational dynamical system for it—comprising a complete set of physical parameters including velocity, acceleration, angular velocity, and angular acceleration. Without any manual annotation, TRACE learns the physical motion laws of 3D scenes from multi-view dynamic videos and accurately extrapolates future frames.

## Background & Motivation

Modeling the geometry, appearance, and physical properties of dynamic 3D scenes is critical for applications in robotics, mixed reality, and embodied AI. Existing approaches fall into two broad categories, each with notable limitations:

**Physics-Informed Neural Networks (PINNs)**: incorporate PDEs as soft constraints in the loss function, but suffer from low training efficiency, poor accuracy near boundaries, and typically require additional annotations such as foreground masks.

**Physics-model encoding methods**: embed specific physical systems (e.g., springs, fluids) into the network, but are restricted to particular material or object types and generalize poorly.

A shared problem across both categories is that they either fail to genuinely learn the physical laws underlying complex motion, or require additional annotations for object types or masks. Meanwhile, existing dynamic 3DGS methods (e.g., DefGS, 4DGS) excel at novel-view synthesis but learn deformation fields that do not encode physical priors—they merely fit pixel-level correlations—and are therefore **entirely incapable of extrapolating future frames**.

Core insight: in scenes with multiple objects or parts, neighboring points may undergo drastically different motions (e.g., at the boundary between two objects moving toward each other), so each 3D point should possess independent dynamical parameters.

## Method

### Overall Architecture

TRACE consists of two core modules and one auxiliary module:
1. **3D Scene Representation Module**: standard 3DGS learning static geometry and appearance at the canonical timestamp $t=0$.
2. **Translational-Rotational Dynamical System Module** (core contribution): learns a complete set of physical parameters for each rigid particle.
3. **Auxiliary Deformation Field**: leverages the deformation network from DefGS/4DGS to stabilize training.

### Key Designs: Translational-Rotational Dynamical System

According to classical mechanics, the motion of any rigid particle $\mathbf{P}$ in 3D space can be decomposed into rotation about a rotation center and translation of that center. For each particle, two groups of physical parameters are learned:

**Group 1 — Rotation center parameters**:
- Center position $\mathbf{P}_c \in \mathbb{R}^3$
- Center velocity $\mathbf{v}_c \in \mathbb{R}^3$
- Center acceleration $\mathbf{a}_c \in \mathbb{R}^3$

**Group 2 — Particle rotation parameters**:
- Rotation vector $\mathbf{w}_p \in \mathbb{R}^3$ (relative to the rotation center)
- Angular acceleration $\boldsymbol{\epsilon}_p \in \mathbb{R}^3$

The combined velocity of the particle is derived as:
$$\mathbf{v}_p^t = \mathbf{w}_p^t \times (\mathbf{P} - \mathbf{P}_c^t) + \mathbf{v}_c^t$$

Since the center velocity and position are coupled, the method instead learns equivalent parameters $\bar{\mathbf{v}}_c^t = \mathbf{v}_c^t - \mathbf{w}_p^t \times \mathbf{P}_c^t$ and $\bar{\mathbf{a}}_c^t$. The entire module is implemented with a simple MLP:
$$\{(\bar{\mathbf{v}}_c^t, \bar{\mathbf{a}}_c^t), (\mathbf{w}_p^t, \boldsymbol{\epsilon}_p^t)\} = f_{trd}(\mathbf{P}, t)$$

**Key advantage**: once the dynamical system parameters are learned at a given time $t$, future particle motion is derived from mechanical laws without requiring additional physical priors.

### Key Designs: Runge-Kutta 2nd-Order Numerical Extrapolation

RK2 is used to propagate from $t'$ to $t = t' + \Delta t$:
1. Compute the equivalent velocity and angular velocity at the intermediate time step.
2. Update position: $\mathbf{x}_t = \mathbf{x}_{t'} + \Delta t (\bar{\mathbf{v}}_c^{mid} + \mathbf{w}_p^{mid} \times \mathbf{x}_{t'})$
3. Compute the incremental rotation matrix $\Delta \mathbf{R}$ via the Rodrigues formula.
4. Scale remains unchanged (rigid-body assumption).

The choice of 2nd order over higher orders is justified by: (1) 2nd-order accuracy is sufficient for short-horizon prediction (millisecond scale); (2) both Newton's first and second laws are captured by 2nd-order relationships; (3) the composition of independent 2nd-order dynamics across a large number of particles is already expressive enough to represent complex deformations.

### Role of the Auxiliary Deformation Field

End-to-end training of the dynamical system module directly is challenging—unstable Gaussian kernel positions in early training hinder optimization. An auxiliary deformation field (e.g., DefGS) is therefore trained in parallel to provide stable particle position inputs to the dynamical system.

### Loss & Training

Standard 3DGS reconstruction loss: $\ell_1 + \ell_{ssim}$, jointly optimizing the canonical Gaussians $G_0$, the deformation field $f_{defo}$, and the dynamical system $f_{trd}$.

## Key Experimental Results

### Main Results: Future Frame Extrapolation (Table 1)

**Dynamic Object dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|:-:|:-:|:-:|
| D-NeRF | 14.660 | 0.737 | 0.312 |
| NVFi | 27.594 | 0.972 | 0.036 |
| DefGS | 19.849 | 0.949 | 0.045 |
| DefGS_nvfi | 28.749 | 0.984 | 0.013 |
| **TRACE (Ours)** | **31.597** | **0.987** | **0.009** |

**Dynamic Indoor Scene dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|:-:|:-:|:-:|
| NVFi | 29.745 | 0.876 | 0.204 |
| DefGS_nvfi | 31.096 | 0.945 | 0.077 |
| **TRACE (Ours)** | **34.824** | **0.965** | **0.054** |

**Dynamic Multipart dataset** (newly proposed; most challenging):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|:-:|:-:|:-:|
| NVFi | 25.235 | 0.955 | 0.046 |
| DefGS_nvfi | 28.455 | 0.979 | 0.017 |
| **TRACE (Ours)** | **33.481** | **0.990** | **0.007** |

### Key Findings

- TRACE substantially outperforms all baselines on 3 of 4 datasets (PSNR gains of 2.8–5.0 dB).
- TRACE also surpasses NVFi on the NVIDIA Dynamic Scene dataset (real-world): 29.341 vs. 28.462.
- DefGS_nvfi—the strongest baseline, which grafts NVFi's velocity field onto 3DGS—is still significantly outperformed by TRACE, demonstrating the superiority of the translational-rotational dynamical system.
- Pure deformation methods (DefGS, 4DGS) perform far worse than physics-constrained methods on the extrapolation task.
- Framework flexibility: replacing the auxiliary deformation field with 4DGS (TRACE_4dgs) also yields strong results.

### Additional Capability: Unsupervised Object/Part Segmentation

Clustering the learned physical parameters naturally segments objects or parts with distinct motion patterns—without any additional annotation.

## Highlights & Insights

1. **Explicit physical parameter learning vs. PINN soft constraints**: TRACE directly learns physical quantities such as velocity and acceleration rather than relying on PDE losses for indirect regularization, achieving higher efficiency and better performance.
2. **Elegance of the rigid-particle assumption**: treating each Gaussian kernel as a rigid particle with size and orientation enables a seamless integration of the natural particle representation of 3DGS with classical mechanics.
3. **The "just enough" philosophy of 2nd-order dynamics**: each particle is governed by only 2nd-order dynamics (acceleration), yet the composition of independently 2nd-order dynamics across a large population of particles can express highly complex scene deformations.
4. **Clever design of the auxiliary deformation field**: it supplements rather than replaces—providing stable inputs during training while inference relies entirely on the physical parameters for extrapolation.

## Limitations & Future Work

- Only 2nd-order dynamics are modeled, which may be insufficient for abrupt motions such as explosions or fractures.
- The rigid-particle assumption (scale invariance) may need to be relaxed for highly deformable soft bodies.
- Extrapolation error grows with prediction horizon; a sliding-window correction mechanism may be needed.
- The performance advantage over baselines diminishes on the real-world NVIDIA dataset compared to synthetic datasets; generalization to more complex real-world scenes requires further validation.

## Related Work & Insights

- **NVFi**: the closest prior work, using PINN losses to learn a velocity field with a NeRF backbone; TRACE supersedes it comprehensively by using explicit physical parameters with a 3DGS backbone.
- **DefGS / 4DGS**: dynamic 3DGS methods that excel at novel-view interpolation but cannot extrapolate; TRACE uses them as auxiliary modules.
- **D-NeRF / HexPlane / TiNeuVox**: dynamic NeRF methods similarly limited to interpolation.
- **PINN series**: physics-informed neural networks (e.g., ScalarFlow) that use PDEs as soft constraints, suffering from low training efficiency.
- **FreeGave**: concurrent work that implicitly fits a velocity network, in contrast to TRACE's explicit learning of motion changes (acceleration/jerk).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The proposed translational-rotational dynamical system constitutes an entirely new paradigm.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Physical modeling is rigorous; RK2 numerical derivation is well-grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets (including a newly proposed one), though broader real-world validation is lacking.
- **Value**: ⭐⭐⭐⭐ — Direct applicability to short-horizon motion prediction in robotic manipulation.
- **Overall Recommendation**: ⭐⭐⭐⭐⭐ — A landmark contribution to physical modeling of dynamic scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)
- [\[CVPR 2026\] Learning a Particle Dynamics Model with Real-world Videos](../../CVPR2026/3d_vision/learning_a_particle_dynamics_model_with_real-world_videos.md)
- [\[ICCV 2025\] Multi-View 3D Point Tracking](multi-view_3d_point_tracking.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](../../NeurIPS2025/3d_vision/mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[ICCV 2025\] From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos](from_gallery_to_wrist_realistic_3d_bracelet_insertion_in_videos.md)

</div>

<!-- RELATED:END -->
