---
title: >-
  [Paper Note] DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics
description: >-
  [ICLR 2026][3D Vision][physics-informed] This paper proposes DiffWind, a physics-constrained differentiable framework that models wind as a grid-based physical field, represents objects as a 3D Gaussian Splatting particle system, simulates wind–object interaction via the Material Point Method (MPM), and incorporates the Lattice Boltzmann Method (LBM) as a physical constraint. The framework jointly reconstructs wind force fields and object motion from video, supports forward simulation under novel wind conditions and wind retargeting, and significantly outperforms existing dynamic scene modeling methods on the newly introduced WD-Objects dataset.
tags:
  - ICLR 2026
  - 3D Vision
  - physics-informed
  - differentiable simulation
  - wind modeling
  - 3D Gaussian Splatting
  - Material Point Method
date: 2026-05-08
content_hash: faae699c61a91c68
---

# DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics

**Conference**: ICLR 2026
**arXiv**: [2603.09668](https://arxiv.org/abs/2603.09668)
**Code**: None (project page mentioned in the paper)
**Area**: 3D Vision / Physical Simulation
**Keywords**: physics-informed, differentiable simulation, wind modeling, 3D Gaussian Splatting, Material Point Method

## TL;DR
This paper proposes DiffWind, a physics-constrained differentiable framework that models wind as a grid-based physical field, represents objects as a 3D Gaussian Splatting particle system, simulates wind–object interaction via the Material Point Method (MPM), and incorporates the Lattice Boltzmann Method (LBM) as a physical constraint. The framework jointly reconstructs wind force fields and object motion from video, supports forward simulation under novel wind conditions and wind retargeting, and significantly outperforms existing dynamic scene modeling methods on the newly introduced WD-Objects dataset.

## Background & Motivation

**State of the Field**: Modeling object dynamics from video observations has been extensively studied, including dynamic scene reconstruction based on NeRF and 3D Gaussian Splatting (3DGS). However, these methods primarily focus on object motion itself or simple interactions, and research on modeling complex deformations driven by invisible external forces—such as wind—remains very limited.

**Limitations of Prior Work**:
   - **Wind is invisible**: Unlike contact forces with well-defined contact points, wind forces are continuously distributed in space and vary over time, making direct observation from video impossible.
   - **Spatiotemporal variability**: Wind force fields vary in both space and time—wind speed and direction may differ across locations within the same scene and evolve temporally.
   - **Complex deformation**: Wind-driven objects (e.g., flags, leaves, cloth) undergo complex non-rigid deformations that are difficult to describe with simple motion models.
   - **Existing dynamic scene reconstruction methods** (e.g., Dynamic 3DGS) only fit appearance changes without modeling the underlying physical forces, and thus cannot generalize to novel wind conditions or support forward simulation.

**Root Cause**: Recovering wind-driven object dynamics from video requires simultaneously estimating the invisible wind force field and the physical response of the object—a highly under-constrained inverse problem. Data fitting alone (e.g., 3DGS) can reconstruct appearance but fails to capture the underlying physical laws, precluding simulation and generalization.

**Paper Goals**: To propose a unified framework that recovers both wind force fields and object motion from video while ensuring physical validity, and that supports downstream applications such as forward simulation and wind retargeting.

**Starting Point**: The paper combines physical simulation primitives (particle system + MPM) with neural rendering (3DGS + differentiable rendering) in a differentiable pipeline to optimize wind force fields from video via backpropagation, while enforcing LBM fluid dynamics constraints to ensure the recovered wind field satisfies physical laws.

**Core Idea**: Differentiable rendering provides appearance supervision + MPM provides physics-based dynamics modeling + LBM provides fluid physics constraints = physically consistent reconstruction of wind–object interaction from video.

## Method

### Overall Architecture
- **Input**: Multi-view (or monocular) video of scenes with wind-driven object motion
- **Output**: (1) 3D reconstruction and motion trajectories of objects; (2) spatiotemporal wind force field distribution; (3) a physical model supporting forward simulation
- **Pipeline**: 3DGS initialization → particle system construction → MPM simulator for wind–object interaction → differentiable rendering supervision → LBM physical constraints → joint optimization of wind field and object motion

### Key Designs

1. **Object Representation: 3DGS-Based Particle System**

    - **Function**: Represents objects in the scene as a set of 3D Gaussian particles, each with position, covariance, color, and opacity.
    - **Mechanism**: Standard 3DGS first reconstructs the object from static frames; Gaussian centers are then extracted as nodes of an MPM particle system. Each particle serves simultaneously as a rendering primitive (for differentiable rendering) and a physics simulation primitive (for MPM time-stepping).
    - **Design Motivation**: 3DGS provides high-quality 3D representation and differentiable rendering. Using its particles directly as carriers for physical simulation avoids the complex meshing steps required by NeRF-based approaches.

2. **Wind Force Field Representation: Grid-Based Physical Field**

    - **Function**: Models wind as a grid-based velocity field in 3D space, where each grid cell stores a wind velocity vector that can evolve over time.
    - **Mechanism**: The wind field is a free variable to be optimized—the magnitude and direction of wind forces at each spatial location and time step are learned by backpropagating through the rendering loss.
    - **Design Motivation**: Compared to parametric wind models (e.g., uniform wind or logarithmic wind profiles), a grid-based representation is more flexible and can capture complex spatial variations. It is also naturally compatible with MPM's grid–particle data structure.

3. **Wind–Object Interaction Modeling: Material Point Method (MPM)**

    - **Function**: Simulates object motion and deformation under wind forces using MPM.
    - **Mechanism**: MPM is a hybrid Lagrangian–Eulerian method—particles (Lagrangian) carry physical quantities (mass, velocity, deformation gradient), while the grid (Eulerian) is used for force computation and momentum solving. Each time step: (1) project particle quantities to the grid; (2) compute internal forces (elastic) and external forces (wind) on the grid; (3) update grid velocities; (4) interpolate updated velocities back to particles and update positions.
    - **Design Motivation**: MPM naturally handles large deformations and complex behaviors such as tearing, and is fully differentiable, enabling end-to-end backpropagation optimization.

4. **Physical Constraint: Lattice Boltzmann Method (LBM)**

    - **Function**: Uses LBM as a physics-based prior to ensure that the optimized wind field satisfies fundamental fluid dynamics laws (a discrete approximation of the Navier–Stokes equations).
    - **Mechanism**: LBM simulates fluid at the mesoscopic level through collision and streaming of particle distribution functions on a lattice, approximating macroscopic fluid behavior. LBM is incorporated as a regularization term in the optimization objective: the optimized wind field must not only match the video through rendering (appearance constraint) but also satisfy LBM fluid consistency (physical constraint).
    - **Design Motivation**: Purely data-driven wind field optimization may yield physically implausible solutions (e.g., violating mass or momentum conservation). LBM constraints reduce the solution space and guide optimization toward physically feasible solutions.

5. **Joint Optimization: Differentiable Rendering + Differentiable Simulation**

    - **Function**: Backpropagates pixel-level rendering loss through an end-to-end differentiable pipeline to wind force field parameters.
    - **Mechanism**: Forward pass: wind field → MPM simulation → particle position update → 3DGS rendering → comparison with video frames. Backward pass: rendering loss → Gaussian parameter gradients via differentiable rendering → wind field gradients via differentiable MPM → wind field update.
    - **Design Motivation**: Making the entire process end-to-end differentiable allows the otherwise unobservable wind force field to be indirectly learned from video supervision signals.

### Loss & Training
- **Rendering Loss**: Standard photometric loss for 3DGS (L1 + SSIM), measuring the discrepancy between rendered images and video frames.
- **LBM Physical Constraint**: Incorporated as a regularization term, penalizing wind fields that deviate from LBM fluid simulation predictions.
- **Joint Optimization Strategy**: 3DGS is first initialized on static frames; the wind field and object deformation are then jointly optimized on dynamic sequences.
- Throughout optimization, wind field parameters and object physical parameters (e.g., elastic coefficients) are updated simultaneously.

## Key Experimental Results

### Dataset: WD-Objects
The paper introduces the WD-Objects dataset, comprising:
- **Synthetic scenes**: Wind-driven object motion generated in a physics simulator (flags, cloth, leaves, etc.) with precise ground-truth wind fields.
- **Real-world scenes**: Videos of wind-driven objects recorded in real-world environments.
- Synthetic data supports quantitative evaluation (with ground truth); real-world data supports qualitative evaluation.

### Main Results

| Task | Metric | Ours | Prev. SOTA | Gain |
|------|--------|------|-----------|------|
| Dynamic reconstruction (synthetic) | PSNR↑ | Significantly higher | Dynamic 3DGS, etc. | Large margin |
| Dynamic reconstruction (synthetic) | SSIM↑ | Significantly higher | Dynamic 3DGS, etc. | Large margin |
| Dynamic reconstruction (synthetic) | LPIPS↓ | Significantly lower | Dynamic 3DGS, etc. | Large margin |
| Wind field estimation (synthetic) | Wind speed error | Physically plausible | No comparable baseline | — |
| Forward simulation | Visual quality | High fidelity | Not supported | — |

Baselines include Dynamic 3DGS, PhysGaussian, and other dynamic scene modeling methods. DiffWind significantly outperforms all baselines in both reconstruction accuracy and simulation fidelity.

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Remove LBM constraint | Decreased physical plausibility of wind field | Without physical constraints, the wind field may violate fluid dynamics |
| Remove MPM (appearance fitting only) | Unable to simulate | Degenerates to pure 3DGS reconstruction, losing physical semantics |
| Different material types | Effective across all | Validates generality across cloth, thin shells, leaves, and other materials |
| Different wind speeds/directions | All recoverable | Validates adaptability of wind field optimization to different wind conditions |

### Key Findings
- **DiffWind significantly outperforms purely data-driven dynamic scene reconstruction methods**: Physical constraints not only improve reconstruction quality but also endow the model with simulation and generalization capabilities.
- **LBM constraints are critical**: Without LBM, the optimized wind field can match the video but is physically implausible (e.g., abrupt wind speed distributions).
- **Forward simulation is a unique advantage**: After reconstruction, wind conditions (direction and intensity) can be modified for forward simulation, generating new object motion sequences—a capability entirely beyond purely data-driven approaches.
- **Wind Retargeting**: Wind fields recovered from one scene can be applied to a different object to generate physically plausible new animations.

## Highlights & Insights

- **Recovering physics from invisible forces**: Wind is a prototypical "invisible force"—unobservable directly, inferable only through its effects on objects. DiffWind elegantly achieves this inference through a differentiable chain (video → rendering → simulation → wind field).
- **Elegant unification of physical simulation and neural rendering**: 3DGS particles serve simultaneously as rendering and simulation primitives; MPM grids carry both the wind field and physical computations, eliminating the overhead of converting between heterogeneous representations.
- **LBM as a "physics regularizer"**: Rather than directly solving Navier–Stokes, the discrete LBM formulation is used as a soft constraint to guide optimization—more flexible than hard constraints while ensuring physical plausibility.
- **Application potential beyond reconstruction**: Forward simulation and wind retargeting make the framework not only an "analysis tool" but also a "creative tool," with potential applications in visual effects and wind simulation for virtual reality.
- **Pioneering problem formulation**: This work is among the first in the 3D vision community to systematically study the recovery of wind–object interaction from video, providing a complete dataset and benchmark.

## Limitations & Future Work

- **Computational cost**: Joint optimization of MPM simulation and differentiable rendering is computationally intensive; optimizing a single scene may require substantial time.
- **Single fluid type**: The current framework models only wind (air flow) and does not address object dynamics driven by other fluids such as water or sand.
- **Topological limitations of objects**: Although MPM handles large deformations, its support for topological changes such as tearing and fracture is limited.
- **Difficulty of evaluation on real scenes**: Real-world videos lack ground-truth wind fields, allowing only qualitative evaluation. Future work could incorporate anemometer data for validation.
- **Multi-object interaction**: The current framework primarily handles the interaction between a single object and wind; occlusion and collision among multiple objects are not fully addressed.
- **Wind field initialization**: The optimization may be sensitive to wind field initialization, requiring a reasonable initial guess to avoid local optima.

## Related Work & Insights

- **Differentiable physical simulation**: Differentiable simulation frameworks such as DiffTaichi and Warp underpin this work. DiffWind combines differentiable simulation with neural rendering for inverse problem solving.
- **3D Gaussian Splatting**: Original 3DGS targets static scenes; Dynamic 3DGS extends it to dynamic scenes but lacks physical modeling; PhysGaussian introduces physics but does not address wind forces.
- **Fluid–Structure Interaction (FSI)**: A classical problem in engineering, but traditional FSI requires known boundary conditions. DiffWind infers boundary conditions (wind field) from video, formulating an inverse FSI problem.
- **Insights**: The combination paradigm of differentiable physics + differentiable rendering can be extended to the recovery of other invisible forces, such as magnetic field-driven or acoustic wave-driven dynamics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)
- [\[AAAI 2026\] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](../../AAAI2026/3d_vision/physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)
- [\[CVPR 2026\] Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields](../../CVPR2026/3d_vision/dynamic_black-hole_emission_tomography_with_physics-informed_neural_fields.md)
- [\[NeurIPS 2025\] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics](../../NeurIPS2025/3d_vision/mpmavatar_learning_3d_gaussian_avatars_with_accurate_and_robust_physics-based_dy.md)
- [\[CVPR 2026\] Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](../../CVPR2026/3d_vision/modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)

<!-- RELATED:END -->
