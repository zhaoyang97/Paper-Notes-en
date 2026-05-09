---
title: >-
  [Paper Note] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics
description: >-
  [NeurIPS 2025][3D Vision][3D human avatar] MPMAvatar integrates a Material Point Method (MPM) physics simulator with 3D Gaussian Splatting rendering. Through an anisotropic constitutive model and a novel collision handling algorithm for mesh-based colliders, it achieves accurate and robust physical animation of loose garments. On ActorsHQ and 4D-DRESS, it outperforms PhysAvatar across both geometry and appearance metrics, achieving a 100% simulation success rate vs. 37.6%, with a per-frame simulation time of only 1.1 seconds.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D human avatar
  - physics-based simulation
  - MPM
  - 3D Gaussian Splatting
  - garment dynamics
date: 2026-05-08
content_hash: 8c3b28c467dff0d1
---

# MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2510.01619](https://arxiv.org/abs/2510.01619)
**Code**: [https://KAISTChangmin.github.io/MPMAvatar/](https://KAISTChangmin.github.io/MPMAvatar/)
**Area**: 3D Vision
**Keywords**: 3D human avatar, physics-based simulation, MPM, 3D Gaussian Splatting, garment dynamics

## TL;DR
MPMAvatar integrates a Material Point Method (MPM) physics simulator with 3D Gaussian Splatting rendering. Through an anisotropic constitutive model and a novel collision handling algorithm for mesh-based colliders, it achieves accurate and robust physical animation of loose garments. On ActorsHQ and 4D-DRESS, it outperforms PhysAvatar across both geometry and appearance metrics, achieving a 100% simulation success rate vs. 37.6%, with a per-frame simulation time of only 1.1 seconds.

## Background & Motivation

**State of the Field**: Creating 3D human avatars from multi-view video is a core problem in computer vision and graphics. Recent methods based on 3D Gaussian Splatting (3DGS) have made significant advances in free-viewpoint rendering. However, physically realistic animation of loose garments (e.g., skirts, trench coats) remains a major challenge.

**Limitations of Prior Work**: Mainstream methods drive garment motion using Linear Blend Skinning (LBS) or pose-dependent geometric corrections, which fail to capture complex deformations such as wrinkles and flowing motion, and severely overfit to training motions. The few works that incorporate physics simulation have critical limitations: Xiang et al. use X-PBD but require time-consuming manual parameter search; PhysAvatar employs the C-IPC simulator, but even minor self-intersections in the driving body mesh—extremely common in practical parametric human estimation—cause complete simulation failure, necessitating manual mesh adjustment. On the appearance side, PhysAvatar relies on mesh-based rendering and cannot capture fine-grained texture.

**Root Cause**: Realistic garment animation requires physics simulation for generalization, yet existing simulators struggle to balance robustness and accuracy—C-IPC offers reasonable accuracy but is extremely brittle, while PBD is fast but physically imprecise. **Core Idea**: The Material Point Method is adopted as the simulation backbone. MPM naturally handles large deformations and complex collisions. The method further tailors the simulator with an anisotropic constitutive model for the physics of cloth and a novel mesh-collider collision algorithm, integrated with 3DGS rendering to achieve comprehensive advantages in accuracy, robustness, and efficiency.

## Method

### Overall Architecture
MPMAvatar consists of three tightly coupled modules: (1) a hybrid avatar representation—triangular meshes with physical parameters for geometry and dynamics, and 3D Gaussian Splats for appearance; (2) a physics dynamics simulator based on a customized MPM—driving garment motion via an anisotropic constitutive model and novel collision algorithm; (3) learning of physical and appearance parameters from multi-view video. The body region is driven by LBS, while the garment region is driven by MPM simulation.

### Key Designs

1. **Anisotropic Constitutive Model**:

    - *Function*: Accurately models the direction-dependent physical behavior of garments—easy stretching along in-plane directions, near-incompressibility in the normal direction.
    - *Mechanism*: The method adopts the anisotropic constitutive model of Jiang et al., reparameterizing the strain energy density $\psi$ via QR decomposition of the deformation gradient into three independent components: $\psi_\text{normal}$ (normal deformation penalty, controlled by stiffness $\kappa$), $\psi_\text{shear}$ (shear penalty, controlled by $\gamma$), and $\psi_\text{in-plane}$ (in-plane deformation, controlled by Young's modulus $E$ and Poisson's ratio $\nu$). Material directions are tracked per particle to differentiate mechanical responses across directions.
    - *Design Motivation*: Standard MPM uses isotropic constitutive models (e.g., Neo-Hookean), which are inadequate for cloth as a codimensional manifold—leading to tearing artifacts (Fig. 4b). The anisotropic model correctly produces fabric-characteristic wrinkles and draping.

2. **Collision Handling Algorithm for Mesh-Based Colliders**:

    - *Function*: Effectively resolves collisions between garments and SMPL-X body meshes to prevent penetration.
    - *Mechanism*: Standard MPM collision handling assumes analytic level sets (e.g., spheres) and cannot handle mesh colliders. A two-stage approach is proposed: (a) *Mesh-to-Grid Transfer*—transferring velocity and normals of each collider face to nearby grid nodes via B-Spline weights; (b) *Relative Velocity Projection*—detecting and removing penetrating velocity components in the collider's reference frame. The complexity is $O(N_f)$, far below the $O(N_\text{grid}^3)$ of analytic level set methods (20K vs. 8M operations).
    - *Design Motivation*: The collider in a physical avatar is an SMPL-X body mesh rather than a simple geometry, requiring support for arbitrary mesh colliders. Furthermore, MPM's feed-forward velocity projection (as opposed to C-IPC's iterative solving) ensures that minor self-intersections in the body mesh do not cause simulation failure.

3. **Quasi-Shadowing + Inverse Physics Learning**:

    - *Function*: Improves rendering realism and enables automatic learning of physical parameters from video.
    - *Mechanism*: Quasi-shadowing uses a neural network to predict a per-Gaussian shading scalar $w_p$ to simulate self-occlusion shadows. For physical parameter learning, $\nu$, $\gamma$, and $\kappa$ are fixed to default values, while Young's modulus $E$, density $\rho$, and rest geometry parameter $\alpha$ (compensating for gravity-induced initial deformation) are optimized end-to-end via finite differences.
    - *Design Motivation*: Shadows are critical to visual realism in physical animation. The rest geometry parameter $\alpha$ addresses a practical problem—canonical meshes recovered from real video are already deformed by gravity, requiring recovery of the unloaded rest shape as the simulation starting point.

### Loss & Training
Physical parameter learning minimizes per-vertex $L_2$ distance between the simulated mesh and the tracked mesh, optimized via finite differences (non-differentiable simulation). Appearance learning minimizes photometric loss between rendered and ground-truth images across all training frames and viewpoints, using the standard 3DGS optimization pipeline.

## Key Experimental Results

### Main Results

| Dataset | Method | CD(×10³)↓ | F-Score↑ | LPIPS↓ | PSNR↑ | SSIM↑ |
|--------|------|----------|---------|--------|-------|-------|
| ActorsHQ | ARAH | 1.12 | 86.1 | 0.055 | 28.6 | 0.957 |
| ActorsHQ | GS-Avatar | 0.91 | 89.4 | 0.044 | 30.6 | 0.962 |
| ActorsHQ | PhysAvatar | 0.55 | 92.9 | 0.035 | 30.2 | 0.957 |
| ActorsHQ | **MPMAvatar** | **0.42** | **95.7** | **0.033** | **32.0** | **0.963** |
| 4D-DRESS | PhysAvatar | 0.37 | 96.6 | 0.022 | 33.2 | 0.976 |
| 4D-DRESS | **MPMAvatar** | **0.33** | **97.2** | **0.018** | **34.1** | **0.977** |

| Method | Simulation Success Rate (%)↑ | Per-Frame Simulation Time (s)↓ |
|------|---------------|-----------------|
| PhysAvatar | 37.6 | 170.0 |
| **MPMAvatar** | **100.0** | **1.1** |

### Ablation Study

| Configuration | CD(×10³)↓ | PSNR↑ | Notes |
|------|----------|-------|------|
| Full (MPMAvatar) | 0.42 | 32.0 | Complete method |
| − Anisotropy | 6.24 | 28.7 | Removing anisotropy → 15× drop in geometric accuracy, tearing artifacts |
| − Physics | 0.69 | 31.0 | Default parameters without learning → suboptimal dynamics |
| − Shadow | — | 31.8 | Removing quasi-shadowing → 0.2 dB drop in PSNR |

### Key Findings
- MPMAvatar outperforms PhysAvatar on both geometry and appearance: CD reduced by 24% and PSNR improved by 1.8 dB on ActorsHQ.
- Simulation robustness is a decisive advantage: PhysAvatar achieves only a 37.6% success rate (C-IPC crashes due to body mesh self-intersections), while MPMAvatar achieves 100%.
- Per-frame simulation speed is 155× faster (1.1s vs. 170s)—MPM's feed-forward projection is far faster than C-IPC's iterative solving.
- The anisotropic constitutive model is critical: removing it causes CD to surge from 0.42 to 6.24, with cloth tearing artifacts.

## Highlights & Insights
- The combination of **classical physics simulation and modern neural rendering** is a high-value research direction—physics guarantees generalization, neural rendering guarantees realism.
- MPM's core advantage over C-IPC lies in its **feed-forward velocity projection**—the absence of iterative solving means the simulation does not crash due to collision detection failures, which is essential for practical applications where body meshes are imperfect.
- **Zero-shot scene interaction generalization** is a unique capability of physics simulation—avatars can naturally interact with unseen chairs or sand, something learning-based methods cannot achieve.
- The rest geometry parameter $\alpha$ is a highly practical design—it resolves the chicken-and-egg problem of learning physics from real video when the canonical pose is already deformed by gravity.

## Limitations & Future Work
- Relighting is not supported, whereas PhysAvatar supports it.
- Non-garment regions (e.g., hair) are still driven by LBS; strand-based simulation could further improve quality.
- Although material parameters are learned via inverse physics, optimization is performed independently per avatar without cross-identity generalization.
- MPM's time step constraints limit simulation accuracy for fast motions.

## Related Work & Insights
- **Engineering insight on simulator selection**: MPM > C-IPC > X-PBD for garment avatar scenarios—robustness and efficiency are the most critical factors, not theoretical accuracy upper bounds.
- **Value of hybrid representation**: Meshes (physics simulation) + Gaussian Splats (rendering) leverage the strengths of both, making them more suitable than purely implicit or purely explicit representations for scenarios requiring simultaneous simulation and rendering.
- **Inverse physics learning**: Learning physical parameters from video is a general paradigm extendable to a broader range of deformable objects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First combination of MPM and 3DGS; customized anisotropic constitutive model and collision handling are solid contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, comprehensive ablations, and zero-shot interaction demonstrations; primary comparisons are limited to PhysAvatar.
- Writing Quality: ⭐⭐⭐⭐ Method description is detailed and figures are clear.
- Value: ⭐⭐⭐⭐⭐ An exemplary work combining physics simulation and neural rendering; substantial gains in robustness and efficiency have strong practical application prospects.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](../../ICLR2026/3d_vision/learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)
- [\[NeurIPS 2025\] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)
- [\[NeurIPS 2025\] Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders](gaussian-augmented_physics_simulation_and_system_identification_with_complex_col.md)
- [\[ICCV 2025\] TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos](../../ICCV2025/3d_vision/trace_learning_3d_gaussian_physical_dynamics_from_multi-view_videos.md)
- [\[NeurIPS 2025\] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)

<!-- RELATED:END -->
