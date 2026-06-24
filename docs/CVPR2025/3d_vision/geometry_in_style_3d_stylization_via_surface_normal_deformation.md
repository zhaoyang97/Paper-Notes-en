---
title: >-
  [Paper Note] Geometry in Style: 3D Stylization via Surface Normal Deformation
description: >-
  [CVPR 2025][3D Vision][Mesh Stylization] Processes text-driven mesh stylization by optimizing the surface normal directions of a triangular mesh, combined with a differentiable ARAP (dARAP) layer to reconstruct vertex positions, enabling expressive geometric deformations while preserving shape identity.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Mesh Stylization"
  - "Surface Normals"
  - "ARAP Deformation"
  - "Text-driven 3D Editing"
  - "Differentiable Geometry"
date: 2026-05-08
content_hash: 339de4601fcd27b8
---

# Geometry in Style: 3D Stylization via Surface Normal Deformation

**Conference**: CVPR 2025  
**arXiv**: [2503.23241](https://arxiv.org/abs/2503.23241)  
**Code**: [https://threedle.github.io/geometry-in-style](https://threedle.github.io/geometry-in-style)  
**Area**: 3D Computer Vision  
**Keywords**: Mesh Stylization, Surface Normals, ARAP Deformation, Text-driven 3D Editing, Differentiable Geometry

## TL;DR

Processes text-driven mesh stylization by optimizing the surface normal directions of a triangular mesh, combined with a differentiable ARAP (dARAP) layer to reconstruct vertex positions, enabling expressive geometric deformations while preserving shape identity.

## Background & Motivation

**Background**: Text-driven 3D shape editing has recently emerged, with mainstream methods utilizing score distillation losses from diffusion models to guide shape deformation. Existing approaches typically represent deformation based on bump maps or Jacobian fields.

**Limitations of Prior Work**: Bump map-based methods (e.g., Text2Mesh) are overly conservative in their deformations, only achieving surface-texture-level modifications. Conversely, Jacobian field-based methods (e.g., TextDeformer, MeshUp) allow too much freedom in deformation; while they can produce large-scale deformations, they easily destroy the identity features of the original shape, introduce artifacts, and require additional L2 regularization constraints.

**Key Challenge**: There exists a fundamental trade-off between "expressiveness" and "identity preservation" in deformation—bump maps are too restrictive, while Jacobian fields are too free. Neither can simultaneously satisfy the requirements of expressive style and identity preservation.

**Goal**: To find a deformation representation that can produce semantically meaningful large-scale deformations (e.g., turning a chair into an origami style) while preserving the structural features of the original shape (e.g., the proportions of the chair legs and armrests).

**Key Insight**: The authors observe that by using surface normals as the driving signal for deformation and reconstructing vertex positions via an As-Rigid-As-Possible (ARAP) solver, the deformation naturally avoids scaling and shearing due to the local rigidity constraints enforced by ARAP. This by construction guarantees identity preservation.

**Core Idea**: Driven by target normals, a differentiable ARAP deformation layer (dARAP) is employed to compress the multiple iterations of classical ARAP into a single local-global step. This layer is embedded into a gradient descent optimization pipeline, cooperating with semantic losses from a diffusion model to achieve text-driven mesh stylization.

## Method

### Overall Architecture

Given an input triangular mesh and a text prompt, the target normal vectors for each vertex are optimized. In each optimization iteration, the local step of dARAP computes the optimal rotation matrix for each vertex neighborhood from the target normals, while the global step solves for the deformed vertex positions via a Poisson equation. The deformed mesh is passed through a differentiable renderer to generate multi-view images, which are fed into a Cascaded Score Distillation (CSD) loss with a DeepFloyd IF diffusion model to align with the text semantics.

### Key Designs

1. **Differentiable ARAP (dARAP) Deformation Layer**:

    - **Function**: A differentiable module that reconstructs deformed vertex positions from target normals.
    - **Mechanism**: Replaces the multiple alternating local-global iterations of classic ARAP with a single compressed step. The local step solves an orthogonal Procrustes problem for each vertex neighborhood—constructing a matrix $X_k$ containing edge vectors and normals, and performing SVD to obtain the rotation matrix $\hat{R}_k = V_k U_k^\top$. The global step fixes the rotation matrices and solves a least-squares Poisson equation $L\hat{\mathcal{V}} = \text{rhs}$ to obtain the deformed vertex positions. The entire process is differentiable and can be directly embedded into neural networks.
    - **Design Motivation**: Classical ARAP requires iterating to convergence, which cannot be efficiently backpropagated. In contrast, dARAP operates inside a larger gradient descent loop, and a single step per iteration is sufficient to achieve high-quality results. A hyperparameter $\lambda$ controls the normal matching intensity, with a default value of 8.

2. **Surface Normal Deformation Representation**:

    - **Function**: Parameterizes the deformation space using target normal vectors per vertex.
    - **Mechanism**: Directly optimizes $|\mathcal{V}| \times 3$ real numbers, which are normalized into unit normals and used as input to dARAP. In the local step, the computation of the rotation matrices balances preserving the original edge directions and aligning with the target normals, weighted by cotangent weights. This ensures that the deformation is constrained to local rigid rotations, preventing scaling and shearing.
    - **Design Motivation**: Compared with the Jacobian representation, the normal representation naturally restricts the deformation space—allowing only rotation but preventing scaling/shearing. This by construction avoids identity destruction without requiring extra regularization losses.

3. **Cascaded Score Distillation (CSD) Vision Loss**:

    - **Function**: Guides the optimization direction of normals so that the deformation matches the style of the text description.
    - **Mechanism**: Utilizes the two-stage (stage 1 and stage 2) diffusion models of DeepFloyd IF. In each epoch, the deformed mesh is rendered from multiple views (batch size of 8 view images) to compute the CSD loss, which is backpropagated to the target normals. The optimization runs for 2,500 epochs using the Adam optimizer with a learning rate of 0.002.
    - **Design Motivation**: Cascaded T2I models generate higher fidelity guidance signals than single-stage models, which, when paired with normal parameterization, yields semantically plausible and detailed stylization.

### Loss & Training

Only the CSD vision loss is used, with no additional identity preservation regularization required. The initial target normals are set to the area-weighted vertex normals of the original mesh. Post-optimization, the strength of the stylization can be adjusted during inference by varying the value of $\lambda$; increasing $\lambda$ yields stronger stylistic features, while decreasing it makes the result closer to the original shape.

## Key Experimental Results

### Main Results

| Method | Mean Triangle Area Ratio ↓ (Ideal 1.0) | Standard Deviation of Triangle Area Ratio ↓ (Ideal 0) |
|------|-------------------------|------------------------|
| TextDeformer | 0.827 | 0.360 |
| MeshUp | 1.288 | 0.363 |
| Geometry in Style (Ours) | **1.080** | **0.233** |

Evaluated on 20 mesh-prompt pairs, the proposed method achieves an area ratio closest to 1.0 and the smallest standard deviation, demonstrating superior preservation of shape identity.

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| Inference $\lambda=8$ (same as training) | Standard stylization effect |
| Inference $\lambda > 8$ | Stronger yet still plausible style, demonstrating the robustness of the method |
| Inference $\lambda < 8$ | Weaker but still visible style |
| Local region stylization (out-of-region rotation set to identity) | Localized deformation without boundary artifacts, demonstrating the regularization effect of dARAP |
| Same style across different domains | Consistently applies Lego style across chairs, animals, and vases |

### Key Findings

- The normal representation restricts the deformation space by design, allowing the method to preserve shape identity without extra identity regularization losses, which is much cleaner than Jacobian-based approaches.
- A larger $\lambda$ can be used during inference than in training to generate more geometrically prominent yet plausible stylizations, showing that the optimized normals possess inherent semantic validity.
- Performs well on both organic surfaces (animals, humans) and man-made objects (chairs, vases). The deformation is part-aware (e.g., creases for a tropical-style chair appear on the seat and backrest rather than the legs).

## Highlights & Insights

- **The Elegance of Normals as Deformation Representation**: Restricting the deformation space to a "rotation-only" subspace guarantees identity preservation by mathematical construction. This avoids the clumsy design of Jacobian methods that require explicit regularization. This concept of "implicit constraint through representation design" is far more elegant than "explicit regularization through loss functions."
- **Discovery of Single-step dARAP**: Classic ARAP requires multiple iterations to converge. However, when embedded within an outer gradient descent loop, a single step is sufficient because the target normals themselves are continuously optimized. This insight could inspire other works requiring embedded differentiable geometry solvers.
- **Adjustable Intensity at Inference**: The optimized normals can be reapplied with different values of $\lambda$, providing user-friendly control over style intensity. This is a feature that MeshUp and TextDeformer lack.

## Limitations & Future Work

- Relies on the cotangent Laplacian, requiring the input to be a manifold mesh with good triangle aspect ratios, which necessitates a remeshing preprocessing step.
- May lead to self-intersections (e.g., lamp rods penetrating each other after rotation); although mitigatable by lowering $\lambda$, this is not fully resolved.
- Topology-preserving: cannot add new parts or change the topology, which limits some stylization possibilities.
- Long optimization time (approx. 2 hours and 15 minutes on a single A40), limiting real-time applications.

## Related Work & Insights

- **vs TextDeformer/MeshUp**: These two methods are based on Jacobian fields, offering high deformation degrees of freedom but easily destroying identity. This work fundamentally solves this issue through normal representation combined with ARAP constraints, achieving better CLIP similarity.
- **vs Text2Mesh**: Text2Mesh performs surface texturing based on vertex displacements, which restricts deformation from producing large structural changes. The proposed deformation can generate global silhouette variations.
- **vs Normal Stylization by Liu & Jacobson**: They utilize spherical normal templates, whereas this work uses text prompts, and the resulting deformation is part-aware (different parts can have different target normals).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of normals and dARAP is ingenious, but the core idea is a clever reorganization of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Qualitative results are rich and diverse, while quantitative metrics (area ratio) are effective but relatively simple.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is written clearly and smoothly, with rigorous and readable mathematical derivations.
- Value: ⭐⭐⭐⭐ Holds practical value for the 3D stylization field, and the dARAP layer is transferable to other geometric tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 4Deform: Neural Surface Deformation for Robust Shape Interpolation](4deform_neural_surface_deformation_for_robust_shape_interpolation.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](../../ICCV2025/3d_vision/tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[CVPR 2025\] Feature-Preserving Mesh Decimation for Normal Integration](feature-preserving_mesh_decimation_for_normal_integration.md)
- [\[ICCV 2025\] Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging](../../ICCV2025/3d_vision/hi3dgen_high-fidelity_3d_geometry_generation_from_images_via_normal_bridging.md)
- [\[CVPR 2025\] Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization](morpheus_text-driven_3d_gaussian_splat_shape_and_color_stylization.md)

</div>

<!-- RELATED:END -->
