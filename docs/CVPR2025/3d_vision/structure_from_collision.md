---
title: >-
  [Paper Note] Structure from Collision
description: >-
  [CVPR 2025][3D Vision][Internal Structure Estimation] This paper introduces a brand-new task, "Structure from Collision" (SfC), which aims to infer the invisible internal structure (such as cavities) of an object by observing its appearance changes during collision. The authors design the SfC-NeRF model to optimize the internal density field under physical constraints, appearance preservation constraints, keyframe constraints, and a volume annealing strategy. The effectivenes…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Internal Structure Estimation"
  - "NeRF"
  - "Collision"
  - "Physical Constraints"
  - "Inverse Engineering"
date: 2026-05-08
content_hash: 1d23bff21ec8f843
---

# Structure from Collision

**Conference**: CVPR 2025  
**arXiv**: [2505.21335](https://arxiv.org/abs/2505.21335)  
**Code**: None (Project Page: [https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/sfc/](https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/sfc/) )  
**Area**: 3D Vision / Neural 3D Representations  
**Keywords**: Internal Structure Estimation, NeRF, Collision, Physical Constraints, Inverse Engineering

## TL;DR

This paper introduces a brand-new task, "Structure from Collision" (SfC), which aims to infer the invisible internal structure (such as cavities) of an object by observing its appearance changes during collision. The authors design the SfC-NeRF model to optimize the internal density field under physical constraints, appearance preservation constraints, keyframe constraints, and a volume annealing strategy. The effectiveness of the method is verified on a dataset containing 115 objects with different structures and materials.

## Background & Motivation

**Background**: Neural 3D representation technologies such as NeRF and 3DGS have been able to accurately estimate 3D object structures from multi-view images. Physics-informed variants (such as PAC-NeRF) further integrate differentiable physics simulation into NeRF, supporting dynamic simulation and system identification (such as identifying physical properties like Young's modulus from video sequences) of continuum materials like elastic materials and fluids.

**Limitations of Prior Work**: All existing methods can only estimate the **visible external structure**, and are powerless regarding the **invisible internal structure** (such as cavities, hollow regions) hidden behind the surface. Even if two objects have completely different internal structures, as long as their appearances are identical, static NeRF will learn the exact same 3D representation. This causes severe issues in applications such as object reproduction in virtual/augmented reality and robotic grasp control.

**Key Challenge**: Internal structures cannot be directly observed but manifest themselves indirectly externally by influencing the physical behavior (such as collision deformation) of objects. However, this is a highly ill-posed problem, as multiple internal structures can produce similar collision behaviors.

**Goal**: To define the SfC task—inferring the internal structure of an object (including the shape, position, and size of cavities) from appearance changes in collision videos.

**Key Insight**: The deformation pattern of an object during collision is influenced by its internal structure (e.g., different cavity positions lead to different depression directions). Assuming material physical properties are known (such as Young's modulus, Poisson's ratio, density, and mass), SfC can be viewed as a complementary problem to PAC-NeRF—PAC-NeRF assumes the interior is filled and optimizes physical properties, whereas SfC assumes physical properties are known and optimizes internal structures.

**Core Idea**: Under known physical properties, optimize the internal volumetric density $\sigma^{G'}(t_0)$ of a voxelized NeRF throughout video sequences, such that the physically simulated collision deformation aligns with the observed video while keeping the external structure unchanged.

## Method

### Overall Architecture

SfC-NeRF employs a two-step optimization. **First step**: Use the multi-view images of the first frame of the video to train a standard voxelized NeRF, learning the filled external structure of the object (at this stage, the interior is solid). **Second step**: Freeze the external structure and optimize the internal volumetric density using the entire collision video sequence. Pipeline of the second step: (1) Obtain the particle set $\mathcal{P}^P(t_0)$ from the voxel grid $\mathcal{F}^{G'}(t_0)$ via G2P transfer; (2) Simulate physical collision using the Differentiable Material Point Method (DiffMPM) to advance particles to each timestep; (3) Transfer back to the grid via P2G and render images for each frame; (4) Compare rendered frames with ground truth frames, backpropagate gradients to update $\sigma^{G'}(t_0)$.

### Key Designs

1. **Physical Constraints (Material Properties + Mass Loss)**:

    - **Function**: Utilize known physical properties to narrow down the solution space of the ill-posed problem
    - **Mechanism**: Material properties (Young's modulus $\hat{E}$, Poisson's ratio $\hat{\nu}$, density $\hat{\rho}$) are explicitly encoded when constructing DiffMPM. The mass constraint is implemented via the loss function $\mathcal{L}_{mass} = \|\log_{10}(m) - \log_{10}(\hat{m})\|_2^2$, where $m = \sum_{p} \hat{\rho} \cdot (\frac{\Delta x}{2})^3 \cdot \alpha_p^P$, representing the sum of all particle masses. A logarithmic scale is used to prioritize matching the order of magnitude.
    - **Design Motivation**: There is a codependency between internal structures and physical properties (such as elasticity)—both hollow structures and soft materials can produce high elasticity. By freezing physical properties, the degrees of freedom are restricted to the internal structure.

2. **Appearance Preservation Constraints (Loss + Training Strategy)**:

    - **Function**: Maintain the external surface learned in the first step from being destroyed while optimizing the internal structure
    - **Mechanism**: **Appearance preservation loss**: An additional pixel preservation loss $\mathcal{L}_{pixel_0}$ enforces first-frame reconstruction quality, while a depth preservation loss $\mathcal{L}_{depth_0}$ maintains the 3D surface shape by comparing horizontal and vertical differences of depth maps between the current model and the pre-optimized model. Differences rather than raw depth are used to mitigate depth estimation errors. **Appearance preservation training**: After completing each video sequence optimization, $\mathcal{F}^{G'}(t_0)$ is re-optimized using only the first frame to restore potentially damaged external structures.
    - **Design Motivation**: Optimizing the internal density field directly should theoretically not alter the appearance of opaque objects, but in practice, the external structure is often accidentally modified due to gradient propagation and optimization dynamics.

3. **Volume Annealing**:

    - **Function**: Search for the global optimum and avoid getting trapped in local optima by repeatedly shrinking and expanding the volume
    - **Mechanism**: Optimization starts from a solid state (the result of the first step) and reduces the volume (creating a cavity) through physics and mass constraints. When the volume reduction proceeds in an incorrect direction, it may trap the optimization in a local optimum. Volume annealing achieves an effect similar to simulated annealing by alternating between volume reduction (normal optimization) and volume expansion (G2P→P2G transfer and replacement of $\mathcal{F}^{G'}$).
    - **Design Motivation**: SfC has multiple feasible solutions (cavities can reside in different positions under the same mass constraint), and starting from a solid state easily traps the optimization in the first found solution. The annealing strategy provides a mechanism to jump out of local optima.

### Loss & Training

The overall loss is formulated as $\mathcal{L}_{full} = \mathcal{L}_{pixel} + \lambda_{mass}\mathcal{L}_{mass} + \lambda_{pres}(\mathcal{L}_{pixel_0} + w_{depth}\mathcal{L}_{depth_0}) + \lambda_{key}\mathcal{L}_{pixel_k}$. Among these, the keyframe loss $\mathcal{L}_{pixel_k}$ assigns extra weight to the frame right after the collision happens (as this frame best reflects deformation differences caused by internal structures). Background masks generated via video matting are also utilized to exclude the static background and focus computation on the target object.

## Key Experimental Results

### Main Results

Cavity size $s_c$ variation experiments (Chamfer Distance ×10³ ↓, average over 5 external shapes):

| Method | $s_c$=0 (Solid) | $s_c$=(1/2)³ | $s_c$=(2/3)³ | $s_c$=(3/4)³ | Average |
|------|----------|----------|----------|----------|------|
| Static (first frame only) | 0.093 | 0.294 | 0.920 | 1.574 | 0.720 |
| GO | 0.091 | 0.301 | 0.941 | 1.586 | 0.730 |
| LPO | 0.092 | 0.284 | 0.841 | 1.406 | 0.656 |
| **SfC-NeRF** | **0.081** | **0.122** | **0.195** | **0.262** | **0.165** |

### Ablation Study

| Configuration | $s_c$=(2/3)³ CD | Average CD | Description |
|------|---------|--------|------|
| SfC-NeRF (full) | 0.195 | 0.165 | Full model |
| w/o mass loss | 0.550 | 0.503 | Without mass constraint, performance drops significantly |
| w/o AP loss | 0.898 | 0.688 | Without appearance preservation loss |
| w/o AP training | 0.332 | 0.335 | Without appearance preservation training |
| w/o keyframe | 0.211 | 0.186 | Without keyframe constraint |
| w/o volume annealing | 0.370 | 0.316 | Without volume annealing |

Cavity position $l_c$ variation experiments (CD ×10³ ↓):

| Method | Left | Right | Up | Down | Average |
|------|------|-------|----|----|------|
| Static | 0.841 | 0.842 | 0.815 | 0.813 | 0.828 |
| LPO | 0.791 | 0.787 | 0.796 | 0.743 | 0.779 |
| **SfC-NeRF** | **0.303** | **0.258** | **0.274** | **0.291** | 0.282 |

### Key Findings

- The mass loss contributes the most—removing it causes the average CD to surge from 0.165 to 0.503, indicating that mass information is a key constraint for narrowing down the solution space.
- Both appearance preservation loss and training strategy are indispensable—relying solely on the loss without the training strategy yields poor results, suggesting that the external structure is indeed prone to being destroyed during optimization.
- Larger cavities lead to harder optimization (from 0.081 to 0.262) because more volume needs to be removed from the solid state.
- Baselines such as GO/LPO can even perform worse than Static (without any optimization), demonstrating that naive video fitting destroys already learned structures.
- Detection of cavity position deviations yields excellent results (visualizations show that SfC-NeRF can capture the direction in which the cavity is biased).

## Highlights & Insights

- **Pioneering new task definition**: SfC is a problem that has never been formally defined before—inferring static invisible structures through dynamic collision observations. This is a significant step for the NeRF field from inferring "structure" to inferring "function/properties".
- **Clever analogy of volume annealing**: Inspired by simulated annealing in optimization theory, volume expansion is implemented via a G2P-P2G loop to escape local optima, without introducing randomness.
- **Comprehensive 115-object dataset**: Systematically covers combinations of 5 external shapes × various cavity sizes/positions/materials, demonstrating a rigorous research paradigm.
- **Key role of physical constraints**: Experiments clearly reveal that pure data-driven methods completely fail to solve the SfC problem without physical constraints.

## Limitations & Future Work

- Assumes physical properties (Young's modulus, Poisson's ratio, density, mass) are known, while in real-world scenarios these properties typically need to be estimated independently.
- Currently utilizes only synthetic data (generated by the MLS-MPM simulator); material modeling errors and observation noise in real collision videos may lead to performance degradation.
- Voxel-based representation based on PAC-NeRF has low efficiency and cannot handle high-resolution scenes.
- Even for the best-performing SfC-NeRF, reconstruction accuracy for large cavities remains limited (CD 0.262 vs solid 0.081), indicating that the intrinsically ill-posed nature of SfC makes precise reconstruction highly challenging.
- Future work can explore other modalities such as acoustics (e.g., tapping sounds) and thermal imaging to aid in inferring internal structures.

## Related Work & Insights

- **vs PAC-NeRF**: Complementary relationship—PAC-NeRF assumes the interior is filled and optimizes physical properties, while SfC-NeRF assumes physical properties are known and optimizes the internal structure. Joint optimization could be explored in the future.
- **vs LPO (Lagrangian Particle Optimization)**: LPO also optimizes particle fields but targets sparse-view completion (external structure), is not designed for internal structures, and lacks key components such as appearance preservation and volume annealing.
- **vs GO (Grid Optimization)**: Naive grid optimization fails completely under SfC scenarios, demonstrating that physical constraints and carefully designed training strategies are necessary.
- This task may find applications in industrial non-destructive testing (such as replacing/supplementing ultrasonic testing) and archaeology (inferring the internal structures of artifacts).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Brand-new task definition; the problem itself is highly innovative and inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 115 objects, multiple variables (size/position/materials), and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous problem definition, clear methodological derivation, and well-organized experiments.
- Value: ⭐⭐⭐⭐ Opens up a new direction, but practical application is limited by the strong assumption of known physical properties.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)
- [\[CVPR 2025\] Dense-SfM: Structure from Motion with Dense Consistent Matching](dense-sfm_structure_from_motion_with_dense_consistent_matching.md)
- [\[CVPR 2025\] MeshArt: Generating Articulated Meshes with Structure-Guided Transformers](meshart_generating_articulated_meshes_with_structure-guided_transformers.md)
- [\[CVPR 2025\] ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)
- [\[CVPR 2025\] MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)

</div>

<!-- RELATED:END -->
