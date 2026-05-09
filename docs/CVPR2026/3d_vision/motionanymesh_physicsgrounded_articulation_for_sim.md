---
title: >-
  [Paper Note] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins
description: >-
  [CVPR 2026][3D Vision][articulated objects] This paper proposes MotionAnymesh, a zero-shot framework that uses SP4D kinematic priors to guide VLMs in eliminating kinematic hallucinations, and employs physics-constrained trajectory optimization to guarantee collision-free articulation. The framework automatically converts static 3D meshes into simulation-ready URDF digital twins directly deployable in physics engines such as SAPIEN, achieving a physical executability rate of 87%—far exceeding existing methods.
tags:
  - CVPR 2026
  - 3D Vision
  - articulated objects
  - digital twins
  - physical constraints
  - VLM reasoning
  - URDF generation
date: 2026-05-08
content_hash: 28ff92af57ee4c5c
---

# MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins

**Conference**: CVPR 2026
**arXiv**: [2603.12936](https://arxiv.org/abs/2603.12936)
**Code**: None (not released)
**Area**: 3D Vision / Articulated Object Modeling / Robot Simulation
**Keywords**: articulated objects, digital twins, physical constraints, VLM reasoning, URDF generation

## TL;DR
This paper proposes MotionAnymesh, a zero-shot framework that uses SP4D kinematic priors to guide VLMs in eliminating kinematic hallucinations, and employs physics-constrained trajectory optimization to guarantee collision-free articulation. The framework automatically converts static 3D meshes into simulation-ready URDF digital twins directly deployable in physics engines such as SAPIEN, achieving a physical executability rate of 87%—far exceeding existing methods.

## Background & Motivation
The vast majority of assets in large-scale 3D repositories (e.g., Objaverse) are static meshes that lack the kinematic structure and part-level physical boundaries required for interaction. Converting static assets into URDF models traditionally demands substantial manual effort. Existing automated pipelines suffer from two fundamental deficiencies: (1) methods relying on 2D-to-3D mask lifting disrupt 3D geometric continuity, producing jagged boundaries and failing to handle self-occluded internal structures; (2) directly applying VLMs for open-vocabulary part decomposition causes models to rely on semantic priors rather than physical constraints, frequently generating kinematic hallucinations when confronted with complex mechanical components that lack explicit semantic names—erroneously merging distinct moving parts or over-segmenting coherent structures. Furthermore, existing joint parameter estimation methods lack rigorous spatial-physical constraints: even when predicted joint axes appear plausible, minor geometric deviations accumulate severely over long-range motion, leading to significant mesh interpenetration, structural detachment, or kinematic freezing.

## Core Problem
How can unstructured static 3D meshes be automatically converted into physically executable articulated digital twins in a zero-shot manner? The core challenges are: (1) how to achieve kinematically aware part segmentation while preserving geometric integrity; and (2) how to estimate accurate joint parameters with strict collision-free articulation guarantees.

## Method

### Overall Architecture
MotionAnymesh comprises three integrated stages: (1) kinematically aware part segmentation—extracting 3D-native geometric primitives and clustering them via SP4D + VLM reasoning; (2) joint estimation and optimization—type-aware geometric initialization followed by physics-constrained trajectory refinement; (3) simulation-ready asset finalization—determining motion limits, preserving textures, and exporting URDF.

### Key Designs

1. **3D-Native Fine-Grained Segmentation**: P3-SAM is applied directly in 3D space to extract low-level geometric boundaries based on spatial concavity and structural connectivity, over-segmenting the mesh into a set of geometrically clean, disjoint primitives $P = \{p_1, \ldots, p_m\}$. These primitives preserve perfect physical boundaries but lack high-level kinematic semantics.

2. **SP4D-Guided Multimodal Clustering**: The over-segmented primitives are clustered into kinematically consistent movable parts. The key innovation is the introduction of explicit kinematic priors from SP4D to anchor VLM reasoning. A single rendered reference image is fed into SP4D, which infers and synthesizes multi-view kinematic segmentation masks indicating coarse functional regions of each movable component. Multi-view primitive images with visual IDs and the SP4D kinematic masks are jointly provided to the VLM, enabling it to cross-reference geometric primitives against physical priors—effectively assembling fragments into functionally consistent kinematic groups following a physical "assembly manual," thereby eliminating kinematic hallucinations.

3. **Type-Aware Kinematic Initialization**: Different geometric strategies are adopted based on joint type:

    - **Revolute-Spin**: The contact point cloud exhibits a disc/annular distribution; PCA is applied to the contact point cloud and the eigenvector corresponding to the smallest eigenvalue is taken as the rotation axis; RANSAC then fits a 2D circle in the normal plane to precisely localize the rotation center.
    - **Revolute-Hinge**: The contact region is distributed longitudinally along the rotation axis; the PCA direction with the largest eigenvalue defines the hinge line, and the geometric centroid serves as the pivot.
    - **Prismatic/Sliding**: PCA of the entire part yields three candidate axes; a normalized dual-penalty trajectory validation mechanism (jointly penalizing collision and derailment) selects the optimal sliding direction.

4. **Physics-Constrained Trajectory Optimization**: Although initialization parameters are reasonable, minor deviations accumulate over long-range motion. A unified surface distance minimization objective is constructed: across a series of discrete virtual motion states, the squared SDF distance of contact-interface points—after rigid-body transformation—relative to the static environment is minimized. The Levenberg–Marquardt algorithm is used for optimization; geometric interpenetration generates strong asymmetric penalties that drive joint parameters to converge onto a fully valid motion manifold.

### Loss & Training
The method is a zero-shot inference framework requiring no training. The core optimization objectives are:
- **Trajectory deviation loss**: $\mathcal{L}_{\text{opt}} = \sum_{\varphi} \sum_{x} \|D_{\text{SDF}}(T(x; v, q, \varphi),\, M_{\text{static}})\|^2$, where $\varphi = \theta$ for revolute joints and $\varphi = d$ for prismatic joints.
- **Prismatic axis selection**: $C(v) = \mathcal{L}_{\text{collide}}(v) + \omega \cdot \mathcal{L}_{\text{derail}}(v)$, where $\mathcal{L}_{\text{collide}}$ is the normalized ratio of interpenetrating points, $\mathcal{L}_{\text{derail}}$ is the mean deviation distance of contact points to the new surface, and $\omega = 20$ balances the magnitude difference.
- **Physical limit estimation**: For revolute joints, collision is detected by incrementally sweeping from $0°$ toward $\pm 180°$; for prismatic joints, collision is detected along the inward direction, while the maximum extension limit along the outward direction is determined by a "contact-loss criterion" triggered when contact area drops to zero.

## Key Experimental Results

| Method | mIoU↑ | Count Acc↑ | Axis Err↓ | Pivot Err↓ | Phys Exec↑ |
|---|---|---|---|---|---|
| PARIS | 0.17 | 0.23 | 1.56 | 1.14 | 11% |
| URDFormer | 0.21 | 0.33 | 1.31 | 1.53 | 21% |
| Articulate-AnyMesh | 0.59 | 0.74 | 0.64 | 0.44 | 35% |
| SINGAPO | 0.52 | 0.66 | 0.73 | 0.57 | 43% |
| Articulate-Anything | 0.47 | 0.61 | 0.86 | 0.64 | 46% |
| **MotionAnymesh** | **0.86** | **0.92** | **0.12** | **0.10** | **87%** |

### Ablation Study
- **Removing SP4D priors (w/o SP4D)**: mIoU drops from 0.86 to 0.68 and Count Acc from 0.92 to 0.81. Pure semantic VLM reasoning leads to severe kinematic hallucinations.
- **Removing trajectory optimization (w/o Opt.)**: Axis Err increases from 0.12 to 0.23 and Pivot Err from 0.10 to 0.27; physical executability plummets from 87% to 65%. Geometric initialization alone provides only coarse orthogonal estimates and produces severe interpenetration during long-range articulation.

## Highlights & Insights
- The problem is elegantly decoupled into boundary extraction and semantic reasoning: 3D-native segmentation ensures geometric cleanliness, while SP4D kinematic priors anchor VLM inference to prevent hallucinations.
- The type-aware initialization strategy is carefully designed: Spin joints use the minimum PCA eigenvector + RANSAC circle fitting; Hinge joints use the maximum PCA eigenvector; Prismatic joints use dual-penalty trajectory validation.
- A physical executability rate of 87% is nearly double that of the strongest baseline (46%), underscoring the importance of physical constraints.
- A Real-to-Sim-to-Real pipeline validates practical deployment value.

## Limitations & Future Work
- Performance depends on the quality of the initial over-segmentation by P3-SAM, which may fail on extremely complex topologies.
- SP4D requires a single rendered image as input, potentially producing inaccurate kinematic masks for objects with unfavorable viewpoints.
- GPT-4o serves as the core VLM, incurring high inference costs that limit scalability to large-scale asset libraries.
- Physical limit estimation relies on simple collision detection without accounting for complex physical interactions such as elastic deformation.
- Only rigid articulation is handled; flexible or deformable parts are not supported.

## Related Work & Insights
- **vs. Articulate-AnyMesh**: The latter relies on 2D-to-3D VLM projection heuristics for joint estimation, making it susceptible to 3D spatial hallucinations. MotionAnymesh operates natively in 3D space with physics-constrained optimization, achieving +0.27 higher mIoU.
- **vs. Articulate-Anything**: The latter depends on retrieval from a predefined CAD library, succeeding on in-domain objects but failing entirely on novel geometries (e.g., robotic arms). MotionAnymesh demonstrates stronger zero-shot generalization.
- **vs. SINGAPO/URDFormer**: Retrieval- or generation-based methods are constrained by template libraries and cannot generalize to open-world shapes.

The SP4D kinematic prior + VLM reasoning paradigm is broadly applicable to other VLM scenarios requiring physical consistency. The physics-constrained trajectory optimization approach (SDF + collision penalty) is a valuable reference for any task involving 3D motion prediction. The evaluation paradigm of zero-shot inference validated by a physics engine is worth adopting.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of SP4D-anchored VLM reasoning and physics-constrained optimization is innovative, with a clear decoupled design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Ablation studies validate core components and Real-to-Sim-to-Real experiments confirm practical utility; however, the dataset scale is not clearly specified.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear, method description is detailed, and mathematical derivations are rigorous.
- **Value**: ⭐⭐⭐ — Focused on 3D articulated object modeling, with limited direct relevance to current research directions, though the physics-constraint + VLM paradigm has general applicability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Particulate: Feed-Forward 3D Object Articulation](particulate_feed-forward_3d_object_articulation.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [\[CVPR 2026\] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)
- [\[CVPR 2026\] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI](wanderland_geometrically_grounded_simulation_for_open-world_embodied_ai.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)

<!-- RELATED:END -->
