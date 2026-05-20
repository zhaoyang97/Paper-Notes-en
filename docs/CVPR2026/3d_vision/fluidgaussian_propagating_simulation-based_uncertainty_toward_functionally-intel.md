---
title: >-
  [Paper Note] FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes FluidGaussian, which guides active view selection in 3D reconstruction using uncertainty metrics propagated through fluid simulation…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "physics-aware reconstruction"
  - "fluid simulation"
  - "active view selection"
  - "uncertainty quantification"
date: 2026-05-08
content_hash: d897df8bd077556e
---

# FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.21356](https://arxiv.org/abs/2603.21356)  
**Code**: [GitHub](https://github.com/delta-lab-ai/FluidGaussian)  
**Area**: 3D Vision / 3D Reconstruction
**Keywords**: 3D Gaussian Splatting, physics-aware reconstruction, fluid simulation, active view selection, uncertainty quantification

## TL;DR

This paper proposes FluidGaussian, which guides active view selection in 3D reconstruction using uncertainty metrics propagated through fluid simulation, yielding reconstructions that are not only visually faithful but also physically plausible under interactive simulation.

## Background & Motivation

**Background**: Current 3D reconstruction methods (NeRF, 3DGS) primarily optimize visual fidelity via photometric losses, achieving photorealistic rendering. Active next-best-view (NBV) methods such as ActiveNeRF and FisherRF select optimal viewpoints through variance reduction or Fisher information.

**Limitations of Prior Work**: These methods focus solely on appearance and neglect physical interaction plausibility. A reconstructed 3D model may be visually perfect yet perform poorly under physical simulation—for example, exhibiting excessive velocity field divergence in fluid simulation, indicating that the underlying geometry is physically unreliable.

**Key Challenge**: A fundamental gap exists between visual fidelity in pixel space (PSNR) and physical interaction fidelity. A model with high PSNR may fail as a digital twin when subjected to forces, fluid coupling, or other physical interactions.

**Goal**: To enable 3D reconstruction to go beyond purely visual cues and capture the physical interactions and functional properties of real-world objects.

**Key Insight**: Fluid simulation is used as a "probe" to expose deficiencies in reconstructed geometry. Fluid–object surface coupling provides dense, surface-wide quality feedback signals.

**Core Idea**: Define a fluid simulation–based uncertainty metric, couple it with existing visual-driven NBV strategies, and select viewpoints that jointly improve visual and physical fidelity.

## Method

### Overall Architecture

FluidGaussian is a plug-and-play module that operates in two stages:
- **Step 1**: An existing visual NBV method (ActiveNeRF / FisherRF) proposes Top-K candidate camera poses.
- **Step 2**: FluidGaussian computes a physical uncertainty score for each candidate pose and selects the viewpoint with the worst physical quality (i.e., the one most in need of improvement).

### Key Designs

1. **Simulation-Induced Uncertainty**:

    - A DFSPH (Divergence-Free SPH) simulator "flushes" the reconstructed object from 5 directions (4 horizontal + 1 top).
    - The velocity divergence of fluid particles is computed as $D_i = |(\nabla \cdot \mathbf{v})_i|$, estimated via SPH-weighted summation:
    $(\nabla \cdot \mathbf{v})_i \approx \sum_{j \in \mathcal{N}_{h_r}(i)} V_j \cdot (\mathbf{v}_j - \mathbf{v}_i) \cdot \nabla W(\mathbf{x}_i - \mathbf{x}_j, h_k)$
    - **Rationale**: DFSPH enforces divergence-free flow by design; numerical errors thus arise primarily at fluid–structure interfaces. Geometrically inaccurate surfaces produce larger interface divergence, making this metric a natural indicator of surface quality.

2. **Per-Gaussian Physical Score Aggregation**:

    - Fluid particle divergence values are aggregated onto nearby 3D Gaussians: $D_g^{(IC)} = \frac{1}{|\mathcal{P}_g^{(IC)}|} \sum_{i \in \mathcal{P}_g^{(IC)}} D_i^{(IC)}$
    - The maximum over all 5 initial conditions is taken: $D_g = \max_{IC} D_g^{(IC)}$
    - **Design Motivation**: Multi-directional flushing eliminates directional bias; the max operator ensures worst-case physical defects are captured.

3. **Physics-Aware View Scoring**:

    - For a candidate camera $\mathbf{T}$, per-Gaussian scores are accumulated with visibility weighting:
    $S(\mathbf{T}) = \sum_{g \in \mathcal{G}_\mathbf{T}} w_g(\mathbf{T}) D_g$
    - The weight $w_g(\mathbf{T}) = \frac{\pi R_g(\mathbf{T})^2}{HW}$ is based on the screen-space projected area.
    - The viewpoint with the highest score (maximum physical uncertainty) is selected as the next acquisition target.
    - **Design Motivation**: This effectively prioritizes reconstruction resources toward regions with the most severe physical defects.

4. **Functionally Critical Region Quantification**:

    - Functionally critical regions (e.g., the aerodynamically important front surface of a vehicle) are identified by counting fluid collision events $|\mathcal{P}_g|$.
    - These regions are used for fine-grained evaluation of reconstruction quality in physically relevant areas.

### Loss & Training

- Base training follows the standard 3DGS photometric loss (L1 + SSIM).
- FluidGaussian does not modify the training loss or model architecture; it only alters the view selection strategy, making it fully plug-and-play.
- Reconstruction begins from 2 initial biased views, with a total budget of 10 views.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Blender | FisherRF | 23.42 | 0.876 | 0.107 |
| Blender | +FluidGaussian | **24.74** | **0.891** | **0.092** |
| DrivAerNet++ | ActiveNeRF | 17.38 | 0.908 | 0.160 |
| DrivAerNet++ | +FluidGaussian | **18.87** | **0.930** | **0.127** |
| MipNeRF360 | FisherRF | 15.32 | 0.471 | 0.456 |
| MipNeRF360 | +FluidGaussian | **15.55** | **0.472** | **0.452** |

Visual PSNR improves by up to +8.6%; velocity field divergence is reduced by up to −62.3%.

### Ablation Study

| Configuration | Observation |
|------|------|
| 5 initial conditions vs. single direction | Multi-directional flushing eliminates bias; a single direction may miss defects in certain regions |
| Functionally critical region PSNR | +7.7% PSNR improvement in fluid-interacting regions |
| High / low Reynolds number | Physical defects are more pronounced under high turbulence, where FluidGaussian yields larger gains |

### Key Findings

- Optimizing for visual fidelity alone systematically underestimates physical quality—visually appealing reconstructions are not necessarily physically plausible.
- Fluid simulation as a probe can expose geometric defects (gaps, self-intersections, thin structures) at fine granularity.
- Improving physical fidelity simultaneously improves visual quality, demonstrating that the two objectives are not in conflict.

## Highlights & Insights

- This work is the first to incorporate fluid simulation into uncertainty quantification for 3D reconstruction, pioneering a "functionally intelligent" reconstruction paradigm.
- The plug-and-play design requires no architectural changes, no modification of training objectives, and only alters view selection.
- The metric design is general—divergence can be replaced by other physical quantities such as vorticity.
- The work opens a new direction toward "simulation-ready" reconstruction for digital twin applications.

## Limitations & Future Work

- Fluid simulation incurs non-trivial computational overhead, requiring 5 SPH runs per NBV evaluation.
- The current formulation considers only incompressible fluids and does not extend to compressible or multiphase scenarios.
- The view budget is fixed at 10; adaptive budget strategies are not explored.
- Only rigid boundary conditions are supported; deformable objects are not addressed.

## Related Work & Insights

- This work bridges two communities: 3D reconstruction (3DGS / NeRF) and physics simulation (SPH fluid dynamics).
- It provides a new perspective for digital twins in engineering and scientific domains—reconstructions must not only "look correct" but also "function correctly."
- The approach suggests that other physical interactions (collision, deformation) could similarly serve as quality signals for reconstruction evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Pioneering introduction of fluid simulation–based uncertainty into 3D reconstruction, defining a new "physics-aware" reconstruction paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets covering synthetic, real-world, and scientific scenarios; large-scale validation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, coherent methodology, and complete physical derivations.
- Value: ⭐⭐⭐⭐ — Significant practical value for digital twin construction and simulation-ready asset creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](../../ICLR2026/3d_vision/peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2026\] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI](wanderland_geometrically_grounded_simulation_for_open-world_embodied_ai.md)

</div>

<!-- RELATED:END -->
