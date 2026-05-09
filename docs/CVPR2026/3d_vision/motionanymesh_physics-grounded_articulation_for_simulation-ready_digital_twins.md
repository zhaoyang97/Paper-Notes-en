---
title: >-
  [Paper Note] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins
description: >-
  [CVPR 2026][3D Vision][Articulated Objects] This paper presents MotionAnymesh, a zero-shot automated framework that converts static 3D meshes into collision-free, simulation-ready articulated digital twins via motion-aware segmentation (SP4D priors + VLM reasoning) and geometry-physics joint optimization for joint estimation, achieving 87% physical executability on PartNet-Mobility and Objaverse.
tags:
  - CVPR 2026
  - 3D Vision
  - Articulated Objects
  - Digital Twins
  - URDF
  - Physics Simulation
  - VLM
date: 2026-05-08
content_hash: 2add9e464faaae3f
---

# MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins

**Conference**: CVPR 2026
**arXiv**: [2603.12936](https://arxiv.org/abs/2603.12936)
**Code**: None
**Area**: 3D Vision
**Keywords**: Articulated Objects, Digital Twins, URDF, Physics Simulation, VLM

## TL;DR

This paper presents MotionAnymesh, a zero-shot automated framework that converts static 3D meshes into collision-free, simulation-ready articulated digital twins via motion-aware segmentation (SP4D priors + VLM reasoning) and geometry-physics joint optimization for joint estimation, achieving 87% physical executability on PartNet-Mobility and Objaverse.

## Background & Motivation

Converting static 3D meshes into interactive articulated assets is critical for embodied AI and robotic simulation. Existing zero-shot approaches suffer from two fundamental flaws: (1) methods relying on 2D-to-3D mask lifting break geometric continuity, resulting in jagged part boundaries and an inability to handle internally occluded structures; (2) when VLMs are used directly for open-vocabulary part decomposition, models rely on semantic priors rather than physical constraints, frequently producing "kinematic hallucinations" when faced with complex mechanical components that lack explicit semantic labels. More critically, existing joint parameter estimation methods lack rigorous spatial-physical constraints — even when predicted joint axes appear plausible, small geometric deviations accumulate severely during long-range actuation, causing significant mesh interpenetration.

## Method

### Overall Architecture

A three-stage pipeline: (1) motion-aware part segmentation — extracting 3D-native geometric primitives and clustering via SP4D motion priors to guide VLM reasoning; (2) joint estimation and optimization — type-aware geometric initialization combined with physics-constrained trajectory optimization to guarantee collision-free kinematics; (3) simulation-ready asset export — determining motion limits, preserving textures, and outputting standard URDF models.

### Key Designs

1. **Motion-Aware Part Segmentation (SP4D-Guided Multimodal Clustering)**: P3-SAM is first applied in 3D-native space to extract fine-grained geometric primitives $\mathcal{P} = \{p_1, \ldots, p_m\}$, ensuring clean geometric boundaries. SP4D-generated multi-view motion segmentation masks are then introduced as explicit motion priors and fed into a VLM alongside rendered primitive visualizations. The VLM cross-correlates geometric primitives with motion regions, assembling fragmented primitives into kinematically consistent functional parts $K_i = \bigcup_{j \in \mathcal{I}_i} p_j$ following a "physical assembly manual." Core motivation: purely semantic VLM reasoning frequently produces hallucinations; SP4D priors anchor reasoning in physical reality.

2. **Type-Aware Joint Initialization**: Different geometric strategies are applied for different joint types:

    - **Revolute-Spin** (e.g., wheels, knobs): PCA is applied to the contact point cloud $S_{contact}$; the eigenvector corresponding to the smallest eigenvalue serves as the rotation axis $\mathbf{v}_{init} = \mathbf{n}$. Contact points are projected onto the 2D plane perpendicular to the axis, and RANSAC fits a 2D circle to determine the pivot:
    $q_{init} = \bar{\mathbf{x}} + x_c \mathbf{b}_1 + y_c \mathbf{b}_2$
    - **Revolute-Hinge** (e.g., door hinges): The contact region distributes longitudinally along the rotation axis; the PCA direction corresponding to the largest eigenvalue yields the rotation axis.
    - **Prismatic joints** (e.g., drawers): PCA on the entire part yields three candidate axes; a normalized dual-penalty validation mechanism selects the optimal sliding direction by jointly evaluating collision penalty $\mathcal{L}_{collide}$ and derailment penalty $\mathcal{L}_{derail}$:
    $\mathcal{C}(\mathbf{v}) = \mathcal{L}_{collide}(\mathbf{v}) + \omega \cdot \mathcal{L}_{derail}(\mathbf{v})$

3. **Physics-Constrained Trajectory Optimization**: Initialization parameters may contain small deviations that accumulate into interpenetration during long-range motion. Joint parameters are refined via unified surface distance minimization using the Levenberg–Marquardt algorithm:
    $\mathcal{L}_{opt}(\mathbf{v}, \mathbf{q}) = \sum_{\phi \in \Phi}\sum_{\mathbf{x} \in S_{contact}}\|\mathcal{D}_{SDF}(\mathcal{T}(\mathbf{x}; \mathbf{v}, \mathbf{q}, \phi), \mathcal{M}_{static})\|_2^2$
   The SDF ensures that moving parts maintain a minimal uniform distance from the static base throughout the full motion range, guaranteeing physically valid, collision-free kinematics.

### Loss & Training

- Segmentation stage: No training required; P3-SAM + SP4D + GPT-4o zero-shot inference.
- Joint estimation: Two stages — PCA/RANSAC geometric initialization → SDF + Nelder–Mead nonlinear optimization.
- Motion limit estimation: Forward simulation with collision detection determines rotation limits; prismatic joints use contact area drop-to-zero detection for disengagement limits.

## Key Experimental Results

### Main Results

| Method | mIoU↑ | Count Acc↑ | Type Err↓ | Axis Err↓ | Pivot Err↓ | Physical Executability↑ |
|--------|-------|-----------|-----------|-----------|-----------|------------------------|
| PARIS | 0.17 | 0.23 | 0.67 | 1.56 | 1.14 | 11% |
| Articulate-Anything | 0.47 | 0.61 | 0.21 | 0.86 | 0.64 | 46% |
| Articulate-AnyMesh | 0.59 | 0.74 | 0.35 | 0.64 | 0.44 | 35% |
| **MotionAnymesh** | **0.86** | **0.92** | **0.08** | **0.12** | **0.10** | **87%** |

### Ablation Study

| Configuration | Key Metrics | Notes |
|---------------|-------------|-------|
| w/o SP4D (VLM semantics only) | mIoU 0.68, Count Acc 0.81 | Severe kinematic hallucinations; over-/under-segmentation |
| SP4D-Guided (Ours) | mIoU 0.86, Count Acc 0.92 | SP4D priors effectively eliminate hallucinations |
| w/o Opt. (initialization only) | Axis Err 0.23, Executability 65% | Small deviations accumulate rapidly during long-range motion |
| Physics-Constrained Opt. | Axis Err 0.12, Executability 87% | Collisions eliminated after optimization |

### Key Findings

- Physical executability is the most critical metric — initialization-only static metrics appear reasonable, yet the large gap from 65% to 87% in dynamic simulation reveals the necessity of physics-constrained optimization.
- SP4D motion priors improve mIoU by 0.18 and Count Acc by 0.11 over purely semantic VLM reasoning.
- Existing retrieval-based methods fail catastrophically on novel open-world geometries.

## Highlights & Insights

- The design philosophy of decoupling "perception" from "actuation" is clear: 3D-native segmentation first ensures geometric purity, then motion priors guide semantic assembly.
- The dual-penalty trajectory validation mechanism (collision + derailment) embodies strong physical intuition — a correct sliding axis should neither cause interpenetration nor derailment.
- Physical executability of 87% is nearly twice that of the strongest baseline (46%), and end-to-end Real-to-Sim-to-Real robotic manipulation validation confirms practical utility.

## Limitations & Future Work

- Relies on GPT-4o as the core VLM, incurring high inference costs; a single complex object may require multiple rounds of VLM calls.
- SP4D infers motion priors from a single image, which may be insufficient for extremely complex or nested structures (e.g., multi-stage gearboxes).
- Flexible joints (e.g., springs, rubber connections) and continuous kinematic chains are not handled.
- Physical limit estimation relies on discrete collision detection step sizes, with accuracy bounded by step granularity.
- Joint types with internal springs or damping cannot be processed.
- Symmetric objects (e.g., double-leaf doors) may have left and right parts incorrectly merged into a single moving component.

## Related Work & Insights

- **vs. Articulate-Anything**: Retrieval-based methods depend on predefined CAD libraries and fail catastrophically on novel open-world geometries; the proposed 3D-native zero-shot approach generalizes far more broadly.
- **vs. Articulate-AnyMesh**: Purely semantic VLM reasoning leads to kinematic hallucinations; this work anchors reasoning in physical reality using SP4D motion priors.
- **vs. PARIS**: PARIS requires multi-state observations (open/closed); this work starts from a single static mesh, imposing lower input requirements.
- The concept of using SP4D motion priors as "physical anchors" for VLM visual reasoning is generalizable to other VLM applications requiring physical awareness.
- The SDF-based trajectory optimization paradigm is applicable to other 3D generation tasks requiring physical compliance guarantees (e.g., furniture assembly, mechanical design verification).
- The comparison between P3-SAM's 3D-native segmentation and 2D-to-3D lifting strongly demonstrates the necessity of operating directly in 3D space.
- The dual-penalty validation mechanism for prismatic joints (collision + derailment) integrates kinematic and geometric constraints in an intuitively appealing design.
- The optional re-texturing module integrating Hunyuan3D demonstrates the framework's strong extensibility.
- The end-to-end Real-to-Sim-to-Real validation (single photo → URDF → policy learning → physical robot deployment) represents the most compelling experimental design.

## Rating

- **Novelty**: ★★★★☆ — The complete pipeline combining SP4D + VLM + physics-constrained optimization is innovative, with the "physics grounding" concept being a notable contribution.
- **Technical Depth**: ★★★★★ — Type-aware initialization (PCA/RANSAC) + SDF trajectory optimization are carefully designed with strong physical intuition.
- **Experimental Thoroughness**: ★★★★★ — Three data sources (PartNet-Mobility / Objaverse / generated assets) + Real2Sim2Real validation.
- **Value**: ★★★★★ — Direct URDF output, 87% physical executability, and high practical value for embodied AI and robotic simulation deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Particulate: Feed-Forward 3D Object Articulation](particulate_feed-forward_3d_object_articulation.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [\[CVPR 2026\] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)
- [\[CVPR 2026\] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI](wanderland_geometrically_grounded_simulation_for_open-world_embodied_ai.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)

<!-- RELATED:END -->
