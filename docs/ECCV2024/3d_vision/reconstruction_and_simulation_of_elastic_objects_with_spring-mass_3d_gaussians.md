---
title: >-
  [Paper Note] Spring-Gaus: Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians
description: >-
  [ECCV 2024][3D Vision][3D Gaussians] This paper proposes Spring-Gaus, which integrates a learnable 3D spring-mass model into 3D Gaussian Splatting to reconstruct the appearance, geometry, and physical dynamics parameters of elastic objects from multi-view videos, supporting future prediction and simulation under different conditions.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussians"
  - "Elastic Objects"
  - "Spring-Mass Model"
  - "System Identification"
  - "Physical Simulation"
date: 2026-05-08
content_hash: ac3e3a0b628558b1
---

# Spring-Gaus: Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians

**Conference**: ECCV 2024  
**arXiv**: [2403.09434](https://arxiv.org/abs/2403.09434)  
**Code**: [https://zlicheng.com/spring_gaus](https://zlicheng.com/spring_gaus)  
**Area**: 3D Vision / Physical Simulation  
**Keywords**: 3D Gaussians, Elastic Objects, Spring-Mass Model, System Identification, Physical Simulation

## TL;DR
This paper proposes Spring-Gaus, which integrates a learnable 3D spring-mass model into 3D Gaussian Splatting to reconstruct the appearance, geometry, and physical dynamics parameters of elastic objects from multi-view videos, supporting future prediction and simulation under different conditions.

## Background & Motivation

**Background**: 3D Gaussian Splatting and its dynamic extensions can reconstruct the temporal variations of object appearance and geometry, but they do not capture physical properties. PAC-NeRF attempts to integrate Material Point Method (MPM) physical priors into NeRF, but assumes a known material model and only handles global physical parameters.

**Limitations of Prior Work**: (1) Dynamic scene reconstruction methods (e.g., D-NeRF, 4D-GS) only fit motion trajectories without understanding physical laws, making them incapable of predicting the future or simulating under new conditions; (2) PAC-NeRF assumes known material types (e.g., elastomers), which is inapplicable to real-world heterogeneous objects; (3) Learning physical parameters for each particle in MPM is computationally prohibitive, and NeRF's implicit grid resolution is limited.

**Key Challenge**: There is a need for a physical dynamics model that is both expressive (capable of modeling complex deformations of heterogeneous elastic objects) and computationally efficient (supporting gradient-based inverse optimization).

**Goal**: To reconstruct the appearance, shape, and physical dynamics parameters of objects from multi-view videos, and to support simulation predictions under different initial conditions and environments.

**Key Insight**: The spring-mass system is a classic physical model that does not assume specific materials. It models various elastic behaviors through learnable topologies and parameters (stiffness, damping) and is naturally differentiable.

**Core Idea**: Use 3D Gaussians to represent appearance and geometry, and keypoint-based sparse anchors + spring systems to represent physical dynamics. Reconstruct the two in a decoupled manner and then combine them into a simulatable 3D object.

## Method

### Overall Architecture
The reconstruction consists of three steps: (1) Reconstructing static 3D Gaussians from the multi-view images of the first frame; (2) Generating sparse anchors via volume sampling and refining the Gaussians; (3) Establishing spring connections between anchors, and optimizing physical parameters (stiffness, damping, mass, and initial velocity) via differentiable simulation and rendering losses.

### Key Designs

1. **3D Spring-Mass Model**:

    - **Function**: Represents the physical dynamics of elastic objects without assuming specific material types.
    - **Mechanism**: Volume sampling is performed on Gaussian centers to generate $N_A$ anchors, each having a mass $m_i$ and velocity $v_i$. They are connected to $n_k$ nearest neighbors via KNN to form a spring network. The spring force is defined as $F_k = -\eta \cdot k_{i,j}(\|x_i - x_{i,j}\| - l_{i,j}) \cdot |\Delta l|^{p_k}$ (representing a non-linear spring when $p_k > 0$). Together with damping forces and gravity, the positions are updated using semi-implicit Euler integration.
    - **Design Motivation**: Compared to MPM, which requires dense particles and known material constitutions, the spring-mass model can model complex deformations of heterogeneous objects using sparse anchors and per-spring learnable parameters, and it is fully differentiable.

2. **Soft Vector**:

    - **Function**: Automatically learns the effective number of spring connections for each anchor.
    - **Mechanism**: Introducing a decay vector $\eta = [\eta_0, \ldots, \eta_{n_k}]$, where close-neighbor spring weights are set to 1, and far-neighbor springs decay based on a learnable parameter $\kappa$ as $\eta_j = \text{clamp}(2 - \exp(\text{softplus}(\kappa))^{j-n_c}, 0, 1)$.
    - **Design Motivation**: If $n_k$ is too large, the object becomes too rigid; if too small, it is too soft. The Soft Vector enables the model to automatically learn the appropriate stiffness, avoiding manual parameter tuning.

3. **Decoupled Reconstruction Pipeline**:

    - **Function**: Decouples appearance/geometry reconstruction from physical parameter optimization to reduce optimization difficulty.
    - **Mechanism**: First, freeze physical parameters to reconstruct static 3D Gaussians $\rightarrow$ extract anchors and refine the Gaussians $\rightarrow$ freeze appearance parameters and optimize physical parameters using differentiable simulation and rendering losses.
    - **Design Motivation**: Joint optimization of both appearance and physical parameters has an excessively large and highly non-convex search space. Decoupling them defines clearer optimization targets for each stage.

### Loss & Training
Static reconstruction: Photometric loss + SSIM + Opacity regularization. Dynamic reconstruction: Multi-view, multi-frame photometric loss backpropagated through differentiable simulation to optimize physical parameters.

## Key Experimental Results

### Main Results

| Scene | Spring-Gaus PSNR | PAC-NeRF PSNR | Remarks |
|------|-----------------|---------------|------|
| Synthetic Elastic Ball | Better | Lower | PAC-NeRF assumes known material |
| Synthetic Heterogeneous Object | Significantly Better | N/A | PAC-NeRF cannot handle heterogeneity |
| Real Rubber Duck | Effective | Limited | Validated on real-world data |

### Ablation Study

| Configuration | Performance | Explanation |
|------|------|------|
| Full Spring-Gaus | Best | Spring-mass + decoupled reconstruction |
| W/o Soft Vector | Drop | Fixed $n_k$ is inflexible |
| Non-linear spring ($p_k>0$) | Improved | Non-linear forces better match real materials |
| Joint optimization (w/o decoupling) | Hard to converge | Decoupled strategy is critical |

### Key Findings
- Although simple, the spring-mass model performs remarkably well on real-world elastic objects, capturing collisions, deformations, and bouncing.
- The Soft Vector effectively and automatically balances the "stiffness-softness" trade-off.
- Forward simulation under unseen conditions (e.g., different initial heights, different gravity directions) is achievable on both synthetic and real-world data.

## Highlights & Insights
- **Wisdom of Model Selection**: Instead of using complex MPM/FEM, a classical spring-mass model is deployed. It is flexible enough (with per-spring learnable parameters to handle heterogeneous materials) and simple enough (fully differentiable and highly efficient).
- Decoupling appearance reconstruction from physics reconstruction is a critical engineering insight, avoiding the challenges of high-dimensional joint optimization.
- The paradigm of inversely learning physical parameters from videos (system identification) can be transferred to robotic object manipulation.

## Limitations & Future Work
- The spring-mass model cannot accurately model materials that require continuum mechanics, such as fluids or cloth.
- It depends on multi-view synchronized videos; single-view scenes would require additional depth estimation.
- Collision handling utilizes simple boundary conditions, showing limited capability in handling complex contact scenarios (e.g., multi-object interaction).
- The choices for the number of anchors and spring connections still rely on empirical heuristics.

## Related Work & Insights
- **vs PAC-NeRF**: PAC-NeRF employs MPM but assumes global material parameters, whereas Spring-Gaus uses the spring-mass model to allow per-spring heterogeneous parameters.
- **vs PhysGaussian**: PhysGaussian uses MPM for forward simulation but does not estimate inverse parameters, while Spring-Gaus completes the entire inverse problem from video to physical parameters.
- **vs DANO**: DANO handles rigid bodies, while Spring-Gaus addresses elastic bodies, which is a more challenging setting.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of spring-mass and 3DGS is simple yet effective, and the Soft Vector design is clever.
- Experimental Thoroughness: ⭐⭐⭐ Synthesized and real-world data are used, but the variety of real-world scenes is limited.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the pipeline diagram is intuitive.
- Value: ⭐⭐⭐⭐ The first to reconstruct simulatable elastic 3DGS objects from videos.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Human Hair Reconstruction with Strand-Aligned 3D Gaussians](human_hair_reconstruction_with_strand-aligned_3d_gaussians.md)
- [\[ECCV 2024\] 3D Reconstruction of Objects in Hands without Real World 3D Supervision](3d_reconstruction_of_objects_in_hands_without_real_world_3d.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](../../CVPR2025/3d_vision/riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[ECCV 2024\] PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects](pisr_polarimetric_neural_implicit_surface_reconstruction_for_textureless_and_spe.md)

</div>

<!-- RELATED:END -->
