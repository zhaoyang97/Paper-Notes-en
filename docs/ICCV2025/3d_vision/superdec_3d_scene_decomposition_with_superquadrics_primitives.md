---
title: >-
  [Paper Note] SuperDec: 3D Scene Decomposition with Superquadric Primitives
description: >-
  [ICCV 2025][3D Vision][Superquadrics] SuperDec is a Transformer-based learning approach that decomposes point clouds into compact sets of superquadric primitives. Trained on ShapeNet…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Superquadrics"
  - "3D scene decomposition"
  - "compact representation"
  - "Transformer"
  - "robot manipulation"
date: 2026-05-08
content_hash: fc658b5b40f9a4b1
---

# SuperDec: 3D Scene Decomposition with Superquadric Primitives

**Conference**: ICCV 2025
**arXiv**: [2504.00992](https://arxiv.org/abs/2504.00992)  
**Code**: [Project Page](https://super-dec.github.io)  
**Area**: 3D Vision
**Keywords**: Superquadrics, 3D scene decomposition, compact representation, Transformer, robot manipulation

## TL;DR

SuperDec is a Transformer-based learning approach that decomposes point clouds into compact sets of superquadric primitives. Trained on ShapeNet, it generalizes to real-world scenes and supports downstream applications including robot manipulation and controllable generation.

## Background & Motivation

3D scene representation is fundamental to computer vision and robotics. While methods such as 3D Gaussian Splatting achieve photorealistic reconstruction, their representations are **memory-intensive and non-compact**, lacking explicit control for spatial reasoning.

Geometric primitive decomposition offers a compact and interpretable alternative, yet existing approaches suffer from notable limitations:

**Learning-based methods** (e.g., SQ [Paschalidou]) require **category-specific training**, encode only global features, and fail to generalize.

**Optimization-based methods** (e.g., EMS) assume hierarchical geometric structure, which is ill-suited for common objects such as tables and chairs.

**Scene-level decomposition** (e.g., DBW) is restricted to a small number of primitives and incurs prohibitive optimization cost (approximately 3 hours).

Superquadrics require only 11 parameters (5 for shape and 6 for pose) to represent a rich variety of shapes, offering greater expressiveness than cuboids (9 + 6 = 15 parameters).

## Method

### Superquadric Parameterization

$$f(\mathbf{x}) = \left(\left(\frac{x}{s_x}\right)^{\frac{2}{\epsilon_2}} + \left(\frac{y}{s_y}\right)^{\frac{2}{\epsilon_2}}\right)^{\frac{\epsilon_2}{\epsilon_1}} + \left(\frac{z}{s_z}\right)^{\frac{2}{\epsilon_1}} = 1$$

Radial distance: $d_r = |\mathbf{x}| \cdot |1 - f(\mathbf{x})^{-\epsilon_1/2}|$

### Feed-Forward Network

The architecture follows a Mask2Former-style Transformer design:

1. **Point encoder (PVCNN)**: Extracts point features $\mathcal{F}_{PC} \in \mathbb{R}^{N \times H}$
2. **Superquadric queries**: Initialized with sinusoidal positional encodings $\mathcal{F}_{SQ} \in \mathbb{R}^{P \times H}$
3. **Transformer decoder**: Iteratively refined via self-attention and cross-attention
4. **Segmentation head**: Predicts a soft assignment matrix $M \in \mathbb{R}^{N \times P}$, where $m_{ij} = \sigma(\phi(\mathcal{F}_{PC}) \cdot \mathcal{F}_{SQ})$
5. **Superquadric head**: Predicts 12 parameters (11 for shape and pose, plus 1 existence probability)

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{rec} + \lambda_{par}\mathcal{L}_{par} + \lambda_{exist}\mathcal{L}_{exist}$$

Reconstruction loss = bidirectional Chamfer distance + normal regularization:
$$\mathcal{L}_{\mathcal{P} \to SQ} = \frac{1}{N}\sum_i\sum_j m_{ij}\min_s d(\mathbf{x}_i, \mathbf{x}'_{js})$$

Compactness loss (0.5-norm encouraging fewer active primitives):
$$\mathcal{L}_{par} = \left(\frac{1}{P}\sum_j\frac{\sqrt{m_j}}{P}\right)^2$$

### Levenberg–Marquardt Refinement

Network outputs serve as initialization for the Levenberg–Marquardt (LM) algorithm, which further refines the superquadric parameters using weighted radial distances as residuals.

### Scene-Level Extension

Instance masks are first extracted using Mask3D → each object is normalized → superquadric decomposition is predicted independently per instance.

## Key Experimental Results

### Quantitative Comparison on ShapeNet

| Method | Primitive Type | L1↓ (in-cat.) | L2↓ (in-cat.) | #Prim.↓ | L1↓ (out-cat.) | L2↓ (out-cat.) |
|--------|---------------|---------------|---------------|---------|----------------|----------------|
| EMS | Superquadric | 5.771 | 1.345 | 5.68 | 5.410 | 1.211 |
| CSA | Cuboid | 5.157 | 0.527 | 9.21 | 4.897 | 0.427 |
| SQ | Superquadric | 3.668 | 0.279 | 10 | 4.193 | 0.354 |
| **SuperDec** | Superquadric | **1.698** | **0.047** | **5.8** | **1.847** | **0.061** |

The L2 error is **6× lower** than SQ while using half the number of primitives.

### Generalization

Trained solely on ShapeNet, the model generalizes without fine-tuning to:
- ScanNet++ real-world indoor scenes
- Replica synthetic scenes

### Key Findings

1. SuperDec's L2 error is only 1/6 that of the previous state of the art, while requiring fewer primitives.
2. Out-of-category generalization is strong, with limited performance degradation across categories.
3. Scene-level decomposition combined with Mask3D effectively handles complete 3D scenes.
4. The method supports downstream applications including robot path planning, grasping, and controllable image generation.

## Highlights & Insights

1. **Adapting supervised segmentation to unsupervised geometric segmentation** — The Mask2Former architecture is repurposed for geometry-based unsupervised decomposition.
2. **Two-stage network–optimization design** — The network provides a strong initialization, which is subsequently refined by LM optimization.
3. **Category-agnostic training** — Joint training across multiple categories, with generalization enabled by local point features.
4. **Practical expressiveness of superquadrics** — Only 11 parameters are required, yet the representational capacity far exceeds that of cuboids.

## Limitations & Future Work

- Generalization relies on regular geometric structure; highly irregular shapes may not be well handled.
- Scene-level decomposition is contingent on the quality of Mask3D instance segmentation.
- The maximum number of primitives $P$ must be specified in advance.

## Related Work & Insights

- **Learning-based methods**: Tulsiani (cuboids), Paschalidou (superquadrics), CSA
- **Optimization-based methods**: EMS, Marching Primitives
- **Scene-level methods**: DBW (Differentiable Blocks World), GES, 3D Convex

## Rating

- Novelty: ⭐⭐⭐⭐ (Two-stage Transformer + LM optimization design)
- Technical Depth: ⭐⭐⭐⭐ (Carefully designed losses, complete optimization module)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-dataset evaluation with downstream application demonstrations)
- Practical Value: ⭐⭐⭐⭐⭐ (Compact representation highly valuable for robotics applications)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Proactive Scene Decomposition and Reconstruction](proactive_scene_decomposition_and_reconstruction.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[ICCV 2025\] Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)

</div>

<!-- RELATED:END -->
