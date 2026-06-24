---
title: >-
  [Paper Note] Differentiable Convex Polyhedra Optimization from Multi-view Images
description: >-
  [ECCV 2024][3D Vision][Differentiable rendering] A differentiable convex polyhedra construction method based on duality transform and three-plane intersection solving is proposed. By bypassing implicit field supervision and directly optimizing gradients using multi-view image losses, high-fidelity convex polyhedral shape representation is achieved.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Differentiable rendering"
  - "Convex polyhedra"
  - "Shape representation"
  - "Duality transform"
  - "Multi-view reconstruction"
date: 2026-05-08
content_hash: ef572395ecc8631f
---

# Differentiable Convex Polyhedra Optimization from Multi-view Images

**Conference**: ECCV 2024  
**arXiv**: [2407.15686](https://arxiv.org/abs/2407.15686)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Differentiable rendering, Convex polyhedra, Shape representation, Duality transform, Multi-view reconstruction

## TL;DR

A differentiable convex polyhedra construction method based on duality transform and three-plane intersection solving is proposed. By bypassing implicit field supervision and directly optimizing gradients using multi-view image losses, high-fidelity convex polyhedral shape representation is achieved.

## Background & Motivation

**Background**: Representing complex shapes by assembling simple geometric primitives (convex polyhedra) is a promising method with wide applications in downstream tasks such as shape parsing, physical simulation, collision detection, and compact mesh representation. Methods like CVXNet and BSPNet represent shapes as the union of multiple convex polyhedra.

**Limitations of Prior Work**: Existing methods (e.g., CVXNet, BSPNet) heavily rely on implicit fields (SDF or occupancy fields) as supervision signals, requiring watertight meshes as training data. The process of pre-processing datasets into discrete sample points with ground-truth implicit values imposes a huge burden on both computational resources and storage. Furthermore, shape composition operations (such as sigmoid approximation for occupancy and softmin/softmax approximation for intersection/union) lead to inaccurate SDFs or biased gradient estimations.

**Key Challenge**: The construction of convex polyhedra (calculating the intersection of hyperplanes) is inherently non-differentiable and cannot be directly used for gradient-based optimization. Existing methods bypass this issue using implicit fields, but introduce significant computational overhead and loss of accuracy.

**Goal**: Eliminate the dependence on implicit fields and achieve differentiable convex polyhedral optimization of purely explicit surface models, using only multi-view images as supervision.

**Key Insight**: Leverage the duality transform from computational geometry to decompose the hyperplane intersection calculation into two steps: "non-differentiable topology discovery" + "differentiable vertex position solving".

**Core Idea**: Determine non-differentiably which three planes form each vertex of the convex polyhedron via the duality transform, and then solve the vertex positions using a differentiable linear equation solver, allowing gradients to propagate from the image loss to the plane parameters.

## Method

### Overall Architecture

Given a set of hyperplane parameters $\theta$, a differentiable convex polyhedron is constructed through the following pipeline:
1. Duality transform maps hyperplanes to points in the dual space
2. Compute the convex hull in the dual space, where each facet of the dual convex hull corresponds to a vertex in the primal space
3. Record which three planes form each vertex by intersection (plane IDs)
4. Compute vertex positions differentiably using a linear equation solver
5. Construct triangular meshes and feed them into a differentiable renderer to compute image loss

### Key Designs

1. **Differentiable Convex Polyhedra Rendering (Core Gradient Chain)**:

    - **Function**: Propagate gradients of the image loss to hyperplane parameters
    - **Mechanism**: Decompose the gradient chain into three parts:
    $\frac{\partial \mathcal{L}}{\partial \theta} = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathcal{I}}}_{\text{图像损失}} \cdot \underbrace{\frac{\partial \mathcal{I}}{\partial \mathcal{V}}}_{\text{可微渲染}} \cdot \underbrace{\frac{\partial \mathcal{V}}{\partial \theta}}_{\text{可微顶点位置}}$
    - **Design Motivation**: Directly computing $\partial \mathcal{I}/\partial \theta$ is intractable (due to the lack of explicit meshes, self-intersection issues, and GPU memory explosion in volume rendering), so gradients are acquired indirectly by constructing explicit meshes.

2. **Hyperplane Intersection Solver via Duality Transform**:

    - **Function**: Determine which three hyperplanes intersect to form each vertex of the convex polyhedron
    - **Mechanism**: Given a hyperplane $a_i^T x \leq b_i$, it is dually mapped to a point $(a_{i_x}/b_i, a_{i_y}/b_i, a_{i_z}/b_i)$. By computing the convex hull in the dual space, each facet of the dual convex hull corresponds to a vertex in the primal space, and the three vertex IDs of the facet correspond to the three intersecting plane IDs.
    - **Design Motivation**: The half-space intersection problem can be elegantly transformed into a convex hull problem in the dual space, which is solved efficiently using the QHull library. This step does not need to be differentiable.

3. **Differentiable Three-Plane Intersection Solving**:

    - **Function**: Compute vertex coordinates differentiably based on the known plane IDs
    - **Mechanism**: Vertex positions are obtained by solving a system of linear equations: $Ax = b$, where the rows of $A$ are the normal vectors of the hyperplanes, and $b$ is the offset vector. PyTorch's linear algebra solver natively supports backpropagation.
    - **Design Motivation**: Decoupling the non-differentiable topology discovery from the differentiable geometric computation is the most critical insight of this work.

4. **Optimization Strategies**:

    - **Persistent Convex**: Constrain all plane offsets $b \in \mathbb{R}^+$ to ensure the origin is always inside the convex polyhedron, avoiding degeneration. Introduce additional translation parameters for flexibility.
    - **Convex Purging**: Remove small convex polyhedra whose volume is below a threshold to improve rendering efficiency.
    - **Plane Purging**: Remove redundant hyperplanes that do not participate in any vertex calculation (planes whose dual vertices fall inside the dual convex hull).
    - **Convex Densification**: Apply Loop subdivision to the convex polyhedron mesh, and replace the original plane parameters with the convex hull equations of the subdivided mesh, enhancing the representation capability in high-curvature regions.
    - **Convex Spawning**: Randomly re-initialize pruned convex polyhedra to avoid falling into local minima.

### Loss & Training

- Supervised only by image $L_1$ loss
- Learning rate: translation of convex polyhedra $1 \times 10^{-2}$, hyperplane parameters $1 \times 10^{-3}$
- Total optimization steps: 20,000, including 10 densification and spawning operations
- Use NVDiffRast as the differentiable renderer
- Each shape is optimized independently (non-learning method)

## Key Experimental Results

### Main Results (ShapeNet Shape Reconstruction)

| Method | L1 CD | L2 CD (×1000) | Normal Consistency |
|------|-------|---------------|-------------------|
| VP (32 cuboids) | 0.048 | 1.861 | 0.738 |
| CVXNet (64 convexes) | 0.024 | 0.526 | 0.891 |
| BSPNet (64 convexes) | 0.027 | 0.699 | 0.745 |
| **Ours (32 convexes)** | **0.022** | **0.592** | **0.925** |

### Multi-view Texture Reconstruction (DTU + NeRF Synthetic)

| Dataset | Method | L2 CD |
|--------|------|-------|
| DTU S40 | DBW | 1.16 |
| DTU S40 | **Ours** | **1.01** |
| NeRF Lego | DBW | 0.91 |
| NeRF Lego | **Ours** | **0.44** |
| NeRF Chair | DBW | 1.07 |
| NeRF Chair | **Ours** | **0.35** |

### Ablation Study

| Configuration | L2 CD | Description |
|------|-------|------|
| 16 Convex | 0.32 | High abstraction, insufficient details |
| 32 Convex | 0.14 | Default configuration |
| 64 Convex | 0.08 | Improved details |
| 128 Convex | 0.05 | Near saturation |
| Without Densification | 0.46 | Surface representation capability drops significantly |
| Without Spawning | 0.54 | Local minima, incomplete reconstruction |

### Key Findings

- Surpasses CVXNet and BSPNet using 64 convex polyhedra with only 32 convex polyhedra
- Particularly pronounced advantages in thin structures and fine details (e.g., the Lamp category)
- Densification and Spawning are crucial for reconstruction quality
- Less effective on objects with large cavities (e.g., cabinets) due to occlusion issues inherent in pure RGB supervision

## Highlights & Insights

- **Clever decoupling of differentiable and non-differentiable parts**: separating topology discovery (which planes intersect) from geometric calculation (where intersection points are), where the former does not need to be differentiable and the latter is naturally differentiable, which is highly elegant
- **No 3D implicit field supervision required**: only multi-view images are needed, significantly reducing data requirements and allowing the training data scale to expand by several orders of magnitude
- **The union of convex polyhedra is naturally handled by the renderer**: z-buffer or ray-object intersection naturally handles union operations without explicit Boolean operations
- **Shape parsing capability**: each convex polyhedron naturally corresponds to a semantic part of the object, which can be used for shape parsing

## Limitations & Future Work

- The detail accuracy of convex polyhedral representation is still inferior to general meshes and implicit representations
- Currently limited to single object reconstruction, not yet extended to full scenes
- Non-static mesh topology makes UV unwrapping impossible, restricting texture mapping to volume textures
- The densification and purging processes are relatively heuristic
- Inherent limitations of relying on differentiable mesh rendering (e.g., zero gradients on silhouette edges caused by self-intersecting triangles)

## Related Work & Insights

- **vs CVXNet/BSPNet**: Both rely on implicit field supervision, requiring watertight meshes and large storage; ours uses image supervision, and each vertex position is an exact three-plane intersection rather than an approximation
- **vs 3DGS**: This work borrows the densification/purging concepts from 3DGS, but changes the representation primitives from Gaussian splats to convex polyhedra, which is more suitable for shape parsing
- **vs DifferentiableBlockWorlds**: DBW uses cuboids and superquadrics, whereas ours uses convex polyhedra, significantly leading on the NeRF Synthetic dataset

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of duality transform and differentiable solving is highly elegant, achieving differentiable convex polyhedra optimization without relying on implicit fields for the first time
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers shape reconstruction, texture reconstruction, and shape parsing, with comprehensive ablation studies
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations and complete method descriptions
- Value: ⭐⭐⭐⭐ Opens up a new direction for image-supervised representation of convex polyhedra shapes, likely to impact downstream applications such as physics simulation and collision detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[ECCV 2024\] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [\[ECCV 2024\] CrossScore: Towards Multi-View Image Evaluation and Scoring](crossscore_towards_multi-view_image_evaluation_and_scoring.md)
- [\[ECCV 2024\] SparseSSP: 3D Subcellular Structure Prediction from Sparse-View Transmitted Light Images](sparsessp_3d_subcellular_structure_prediction_from_sparse-view_transmitted_light.md)

</div>

<!-- RELATED:END -->
