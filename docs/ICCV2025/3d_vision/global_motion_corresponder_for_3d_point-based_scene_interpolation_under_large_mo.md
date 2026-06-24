---
title: >-
  [Paper Note] Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion
description: >-
  [ICCV 2025][3D Vision][Scene interpolation] This paper proposes the Global Motion Corresponder (GMC), which learns unary potential fields that map 3D Gaussians from two time steps into a shared canonical space, enabling robust scene interpolation and extrapolation under large motion.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Scene interpolation"
  - "large motion"
  - "3D Gaussian Splatting"
  - "SE(3) transformation"
  - "semantic correspondence"
date: 2026-05-08
content_hash: f216739be6f311bb
---

# Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion

**Conference**: ICCV 2025
**arXiv**: [2508.20136](https://arxiv.org/abs/2508.20136)  
**Code**: [Project Page](https://junrul.github.io/gmc/)  
**Area**: 3D Vision
**Keywords**: Scene interpolation, large motion, 3D Gaussian Splatting, SE(3) transformation, semantic correspondence

## TL;DR

This paper proposes the Global Motion Corresponder (GMC), which learns unary potential fields that map 3D Gaussians from two time steps into a shared canonical space, enabling robust scene interpolation and extrapolation under large motion.

## Background & Motivation

Dynamic scene interpolation reconstructs continuous motion from two sets of discrete multi-view frames and represents a fundamental challenge in computer vision. Existing methods rely on a critical assumption: **motion between adjacent time steps is sufficiently small** that displacements can be approximated by local linear models.

Under large motion, this assumption breaks down fundamentally:

**Local neighborhood search fails** — a point's local neighborhood becomes unreliable as objects have moved far from their original positions.

**Global matching ambiguity** — a single point may have multiple plausible correspondences, making point-to-point matching ill-defined.

**Cross-matching** — naive nearest-neighbor matching produces criss-cross correspondences that cannot be used for scene interpolation.

Existing methods fall into two categories: deformation field methods (4DGS, Deformable-3DGS) and iterative refinement methods (Dynamic Gaussian, PAPR In Motion), both of which fail under large motion.

## Method

### Mechanism

The paper replaces direct point-to-point matching with **unary potential fields** that learn SE(3) transformations for each time step, mapping both sets of Gaussians into a shared canonical space:

$$\underset{\hat{\boldsymbol{\mu}}^{(0)}_i}{\underbrace{\boldsymbol{R}^{(0)}_i\boldsymbol{\mu}_i^{(0)}+\boldsymbol{t}^{(0)}_i}} = \underset{\hat{\boldsymbol{\mu}}^{(1)}_j}{\underbrace{\boldsymbol{R}^{(1)}_j\boldsymbol{\mu}_j^{(1)}+\boldsymbol{t}^{(1)}_j}}$$

### Unary Potential Field Design

$\mathcal{F}(\tilde{\boldsymbol{f}}, \boldsymbol{\mu}) = (\boldsymbol{R}, \boldsymbol{t})$

The network takes PCA-projected DINO semantic features and 3D coordinates as input, and outputs SE(3) transformations. The inductive bias of MLPs ensures smooth outputs: semantically similar points naturally predict similar transformations. Independent MLPs ($\mathcal{F}_0, \mathcal{F}_1$) are used for each time step.

### Energy Loss

$$E_{i,j} = w_c\|\boldsymbol{c}_i - \boldsymbol{c}_j\|_2^2 + w_f\|\boldsymbol{f}_i - \boldsymbol{f}_j\|_2^2 + w_\mu\|\hat{\boldsymbol{\mu}}_i - \hat{\boldsymbol{\mu}}_j\|_2^2$$

A key distinction is the use of **spatial distances in canonical space** rather than in the original Euclidean space.

A bidirectional loss ensures all points find correspondences:
$$\mathcal{L}_E = \sum_{g_i \in \mathcal{G}_0} \min_{g_j \in \mathcal{G}_1} E_{i,j} + \sum_{g_j \in \mathcal{G}_1} \min_{g_i \in \mathcal{G}_0} E_{j,i}$$

### Local Isometry Loss

$$\mathcal{L}_{iso} = \frac{1}{kN}\sum_{g_i}\sum_{g_j \in NN_i}\left|\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2^2 - \|\hat{\boldsymbol{\mu}}_i - \hat{\boldsymbol{\mu}}_j\|_2^2\right|$$

This term preserves local geometric relationships and promotes locally rigid transformations. The total loss is: $\mathcal{L} = \mathcal{L}_E + \alpha \mathcal{L}_{iso}$

### Joint Refinement and Degeneration Prevention

After training, GMC and the Gaussian sets are jointly updated using rendering loss for further optimization. Dropout is applied to the positional input $\boldsymbol{\mu}$ to prevent the degenerate solution in which all points collapse to the origin.

## Key Experimental Results

### Quantitative Comparison for Scene Interpolation (SI-FID↓)

| Method | Ball | Boat | Butterfly | Car | Dolphin | Knight | Microwave | Seagull |
|--------|------|------|-----------|-----|---------|--------|-----------|---------|
| 4DGS | – | 328.8 | – | 460.0 | – | – | 258.2 | 294.0 |
| Deformable-3DGS | – | 811.1 | – | 800.1 | – | – | 709.6 | 633.6 |

(Note: "–" indicates complete failure to produce reasonable renderings.)

### Extrapolation Capability

GMC supports motion extrapolation, whereas all baseline methods cannot. It can predict motion beyond the range defined by the two given time steps.

### Key Findings

1. Under large-motion scenarios, existing deformation-field and iterative-refinement methods fail entirely.
2. Semantic features (DINO) are critical for establishing initial "soft" correspondences.
3. The bidirectional loss is necessary; a unidirectional loss leaves some points without correspondences.
4. The local isometry weight should be gradually increased from zero, first establishing correspondences before enforcing rigidity.

## Highlights & Insights

1. **Problem redefinition** — Large-motion scene interpolation is reformulated as a global correspondence establishment problem.
2. **Effective use of canonical space** — Mapping both Gaussian sets into a shared space avoids direct point-to-point matching.
3. **Effective exploitation of DINO semantics** — Semantic features facilitate meaningful initial matching under large motion.
4. **Extrapolation as a byproduct** — Motion extrapolation is naturally supported without additional design.

## Limitations & Future Work

- Requires multi-view images at each time step to reconstruct 3DGS inputs.
- Assumes approximately locally rigid motion throughout the scene.
- May not generalize to scenes with highly complex topological changes, such as fluid dynamics.

## Related Work & Insights

- **Deformation field methods**: 4DGS, Deformable-3DGS
- **Iterative refinement methods**: Dynamic Gaussian, PAPR In Motion
- **Visual feature correspondence**: DINO, Zero-Shot 3D Shape Correspondence
- **Point cloud interpolation**: Traditional methods similarly fail under large motion

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The canonical space + unary potential field design is highly elegant)
- Technical Depth: ⭐⭐⭐⭐ (Energy function and individual loss designs are well-motivated)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Demonstrates both interpolation and extrapolation capability)
- Value: ⭐⭐⭐⭐ (Addresses a practical bottleneck in large-motion scene settings)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)
- [\[ICCV 2025\] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)
- [\[ICCV 2025\] Sequential Gaussian Avatars with Hierarchical Motion Context](sequential_gaussian_avatars_with_hierarchical_motion_context.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)

</div>

<!-- RELATED:END -->
