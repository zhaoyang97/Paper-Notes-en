---
title: >-
  [Paper Note] Hierarchically Structured Neural Bones for Reconstructing Animatable Objects from Casual Videos
description: >-
  [ECCV2024][Robotics][Animatable 3D Reconstruction] The Hierarchical Neural Bones (HSNB) framework is proposed, which decomposes object motion in a coarse-to-fine manner using a tree-structured bone system to reconstruct high-quality animatable 3D models from casual videos.
tags:
  - "ECCV2024"
  - "Robotics"
  - "Animatable 3D Reconstruction"
  - "Neural Bones"
  - "Hierarchical Deformation"
  - "NeRF"
  - "Manipulation"
date: 2026-05-08
content_hash: de261622508f04c8
---

# Hierarchically Structured Neural Bones for Reconstructing Animatable Objects from Casual Videos

**Conference**: ECCV2024  
**arXiv**: [2408.00351](https://arxiv.org/abs/2408.00351)  
**Code**: [subin6/HSNB](https://github.com/subin6/HSNB)  
**Area**: Robotics  
**Keywords**: Animatable 3D Reconstruction, Neural Bones, Hierarchical Deformation, NeRF, Manipulation

## TL;DR

The Hierarchical Neural Bones (HSNB) framework is proposed, which decomposes object motion in a coarse-to-fine manner using a tree-structured bone system to reconstruct high-quality animatable 3D models from casual videos.

## Background & Motivation

Reconstructing animatable 3D models from casually captured videos has significant application value (e.g., film and television, mixed reality, gaming). However, existing methods suffer from two main issues:

1. **Prior-template-dependent methods** (such as skeletons, 3D human body models) require a large amount of 3D scanning data or annotations, lacking generalization to arbitrary objects.
2. **Template-free methods** (such as BANMo), although generalizable by using Gaussian ellipsoids as control points, lack skeletal structure, scattering across the object surface without considering motion granularity, which leads to:
    - Lack of association between bones, making user editing and control difficult.
    - Requirement of a large number of input videos to produce reasonable results.
    - Room for improvement in reconstruction quality.

## Core Problem

How to learn a set of hierarchically structured control points without relying on any prior structural knowledge, to accurately capture object motion, improve reconstruction quality, and provide users with an intuitive, easily editable 3D model interface?

## Method

The overall architecture is built upon BANMo, with key improvements in the **hierarchical deformation model** and **bone occupancy function**.

### 1. Base Framework (BANMo)

- **Canonical Model**: Uses NeRF to represent the shape and appearance of the object, outputting color and SDF values.
- **Deformation Model**: Uses Gaussian ellipsoid bones and Linear Blend Skinning (LBS) to deform 3D points from each frame to the canonical space.
- All components are jointly optimized through differentiable volume rendering.

### 2. Hierarchical Neural Deformation Model

**Core Idea**: Organize bones into a tree structure, where parent bones capture coarse motion of large areas and child bones represent fine-grained motion of smaller parts.

- **Skeletal Hierarchy Formulation**: The bone transformation at depth $d$ is obtained recursively by left-multiplying all parent transformations:
$$T^d = \hat{T}^1 \hat{T}^2 \cdots \hat{T}^{d-1} \hat{T}^d$$
Child bones are defined in the local coordinate system of parent bones, naturally inheriting parent movements.

- **Neural Bone Representation**: Each bone consists of a rotation $R$, a center $\mathbf{t}$ (varying per frame), and a shared scale $\mathbf{s}$. Different depths use independent MLPs $f^d$, taking parent bone embeddings and root embeddings as input.

- **Skinning Weights**: Computed based on the Mahalanobis distance between each 3D point and the leaf bones, combined with the delta weights predicted by MLPs for LBS deformation.

- **Coarse-to-Fine Optimization**: Depth-1 bones (coarse motion) are optimized first, and child bones are progressively added during training to capture finer motions. Initial number of bones: 5 for animals, 6 for humans; each bone subsequently splits into 2 child bones.

### 3. Bone Occupancy Function

To address the under-constrained bone position and shape issues, bones are regularized to the centers of object parts.

- **Bone Occupancy**: $g_b(\mathbf{x}) = d_M(\mathbf{x}, b) - \gamma$, where values are negative inside the bones and positive outside.
- **Bone Mask Loss** $\mathcal{L}_{bone}$: Render public 2D masks by aggregating occupancies of all bones, and compare with ground-truth foreground masks to encourage alignment between bone shapes and the object.
- **Overlap Loss** $\mathcal{L}_{overlap}$: Restricts each surface point to be covered by no more than $\lambda$ bones to prevent excessive overlap.
- **Coverage Loss** $\mathcal{L}_{cover}$: Ensures that each bone occupies a minimal proportion of the surface area, preventing bone degradation.

### 4. Overall Loss Function

$$\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{cycle} + \mathcal{L}_{bone} + \mathcal{L}_{overlap} + \mathcal{L}_{cover}$$

### 5. Manipulation Method

Once optimization is finished, the canonical mesh is extracted. Users can control large-scale motion by adjusting parent bones, achieve fine-grained control by adjusting child bones, and interactively add or delete control points.

## Key Experimental Results

| Method | Eagle CD↓ | Eagle F2↑ | Swing CD↓ | Swing F2↑ | Samba CD↓ | Samba F2↑ |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| ViSER | 19.22 | 24.76 | 16.29 | 19.95 | 23.28 | 22.47 |
| BANMo (Re-implemented) | 4.66 | 81.44 | 7.33 | 64.88 | 7.22 | 64.99 |
| CAMM* | 4.50 | 81.21 | 9.02 | 56.00 | 7.50 | 62.17 |
| RAC* | - | - | 6.10 | 70.33 | 6.63 | 67.71 |
| **Ours** | **4.64** | **81.59** | **7.11** | **65.88** | **6.15** | **72.07** |

- Outperforms all template-free methods across all datasets, and outperforms RAC (which requires a skeleton prior) on Samba.
- Achieves comparable results on the Eagle dataset using only 10 leaf bones (compared to BANMo's 25+).
- Fully leading in neural rendering quality: PSNR 31.74 / SSIM 0.942 on Samba (compared to BANMo's 30.72 / 0.916).

## Highlights & Insights

1. **Prior-free Hierarchical Structure Learning**: Discovers the part hierarchy and motion correlation of objects completely unsupervised from videos, without requiring predefined skeletons or templates.
2. **Coarse-to-Fine Motion Decomposition**: The tree-structured skeletal system provides multi-granularity motion representations, improving reconstruction quality and enhancing interpretability.
3. **Bone Occupancy Regularization**: Inspired by part-based generative models, it ensures that bones align with object shapes using mask, overlap, and coverage loss terms.
4. **Fewer Control Points, Better Performance**: Uses fewer bones to achieve or even exceed benchmarks that utilize a large number of control points.
5. **User-Friendly Manipulation Interface**: Hierarchical bones support coarse-to-fine control, enabling users to interactively add or remove control points.

## Limitations & Future Work

1. Training takes a long time (20k iterations per depth, approximately 3 hours per stage on 2x RTX 3090), resulting in significant cumulative training time.
2. The number of child bones (2 per parent bone) is manually specified, without adaptive adjustment according to motion complexity.
3. Only evaluated on humans and animals; applicability to objects with large topological changes (e.g., clothes, fluids) remains unknown.
4. Relies on preprocessing (PointRend segmentation, VCN optical flow, CSE feature), where cascading errors can affect results.
5. More efficient representation methods, such as 3D Gaussian Splatting, could replace NeRF to further improve efficiency.

## Related Work & Insights

| Dimension | BANMo | CAMM | RAC | Ours (HSNB) |
|------|-------|------|-----|-------------|
| Prior Knowledge | None | Requires RigNet skeleton | Requires predefined skeleton | None |
| Control Point Structure | Unstructured scattering | Kinematic chain | Fixed skeleton | Hierarchical tree structure |
| Motion Decomposition | None | Limited | Limited | Coarse-to-fine multi-granularity |
| Number of Bones | 25 | 25+ | Defined by category | Minimum of 10 leaf bones |
| Controllability | Difficult | Good | Good | Intuitive & interactive addition/deletion |
| Generalization | Any object | Requires skeleton matching | Single category | Any object |

## Related Work & Insights

- The concept of **Bone Occupancy Function** can be extended to other tasks requiring part alignment (such as part-aware human reconstruction).
- Hierarchical control points can be combined with 3D Gaussian Splatting to build more efficient animatable reconstruction pipelines.
- Coarse-to-fine training strategies have reference value for other hierarchical representation learning tasks (e.g., scene-level hierarchical representations).
- The overlap/coverage loss designs in the bone occupancy function can be applied to regulate other primitive-based methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of hierarchical bone structure and bone occupancy regularization is quite novel in template-free animation reconstruction.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive quantitative and qualitative comparisons across multiple datasets, including ablation studies and manipulation demos.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — Valuable in both utility (manipulation interface) and academic contribution (hierarchical motion decomposition).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](../../NeurIPS2025/robotics/spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[ECCV 2024\] GraspXL: Generating Grasping Motions for Diverse Objects at Scale](graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)
- [\[NeurIPS 2025\] Asymptotically Stable Quaternionic Hopfield Structured Neural Network with Supervised Projection-based Manifold Learning](../../NeurIPS2025/robotics/asymptotically_stable_quaternion-valued_hopfield-structured_neural_network_with_.md)
- [\[CVPR 2026\] Contact-Aware Neural Dynamics](../../CVPR2026/robotics/contact-aware_neural_dynamics.md)
- [\[ICML 2026\] Neural Low-Discrepancy Sequences](../../ICML2026/robotics/neural_low-discrepancy_sequences.md)

</div>

<!-- RELATED:END -->
