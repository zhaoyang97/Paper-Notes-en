---
title: >-
  [Paper Note] CUBE: Representing 3D Faces with Learnable B-Spline Volumes
description: >-
  [CVPR 2026][3D Vision][B-spline volumes] This paper proposes CUBE (Control-based Unified B-spline Encoding), a hybrid geometric representation combining B-spline volumes with learnable high-dimensional control features.…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "B-spline volumes"
  - "face representation"
  - "scan registration"
  - "local control"
  - "geometry editing"
date: 2026-05-08
content_hash: feb6c5823e36bd3c
---

# CUBE: Representing 3D Faces with Learnable B-Spline Volumes

**Conference**: CVPR 2026
**arXiv**: [2604.12894](https://arxiv.org/abs/2604.12894)  
**Code**: None  
**Area**: 3D Vision / Face Reconstruction
**Keywords**: B-spline volumes, face representation, scan registration, local control, geometry editing

## TL;DR

This paper proposes CUBE (Control-based Unified B-spline Encoding), a hybrid geometric representation combining B-spline volumes with learnable high-dimensional control features. Through a two-stage decoding pipeline (B-spline basis interpolation followed by a lightweight MLP residual), CUBE enables editable, high-fidelity 3D face reconstruction and scan registration.

## Background & Motivation

**Background**: 3D face representation is dominated by three paradigms: 3D Morphable Models (3DMMs) provide compact, disentangled linear spaces but lack fine detail; nonlinear neural models improve flexibility but sacrifice interpretability and local control; implicit representations offer high detail but lack semantic correspondence and require costly isosurface extraction.

**Limitations of Prior Work**: 3DMMs are constrained by fixed topology and low-dimensional parameter spaces, making them unable to capture subject-specific high-frequency details. Neural models lack local editing capability. Implicit models are incompatible with standard graphics pipelines.

**Key Challenge**: Local controllability, geometric expressiveness, and computational efficiency are inherently difficult to achieve simultaneously in a single representation.

**Goal**: To design a hybrid face representation that combines the local control properties of B-splines with the expressive power of neural networks.

**Key Insight**: Replace the conventional 3D control points of B-spline volumes with high-dimensional learnable control features, and supplement high-frequency detail via a lightweight MLP.

**Core Idea**: A high-dimensional control feature lattice (e.g., $8\times8\times8$) defines a continuous mapping from the parametric domain to Euclidean space; the B-spline basis provides local support, enabling local editing.

## Method

### Overall Architecture

CUBE is parameterized by a high-dimensional control feature lattice. Given 3D coordinates on a fixed template mesh, B-spline bases locally blend the control features to produce high-dimensional vectors, whose first three dimensions define a base mesh position; the full feature vector is then passed to a lightweight MLP to predict a residual displacement. The output is a 3D surface with dense semantic correspondence.

### Key Designs

1. **High-Dimensional Control Feature Lattice**:

    - **Function**: Parameterize 3D face shape with a compact set of lattice points in place of a dense mesh.
    - **Mechanism**: Whereas conventional B-spline volumes use 3D control points, CUBE replaces them with high-dimensional (e.g., 32-dimensional) control features. B-spline bases locally blend the neighboring control features at each query point to produce a high-dimensional feature vector, preserving the local support property of B-splines — modifying a single control feature affects only its local region.
    - **Design Motivation**: Standard B-spline 3D control points lack the expressive capacity to represent complex face shapes with a small number of lattice points.

2. **Two-Stage Decoding**:

    - **Function**: Capture both global shape and local detail.
    - **Mechanism**: The first three dimensions of the blended high-dimensional feature vector directly define a coarse base mesh (global shape), while the full feature vector is fed to a lightweight MLP to predict residual displacements from the base shape (high-frequency detail).
    - **Design Motivation**: B-spline bases are inherently smooth and ill-suited to representing high-frequency geometry. The MLP compensates for this limitation while preserving local support, since its inputs are derived from locally blended features.

3. **Transformer-Based Encoder**:

    - **Function**: Predict CUBE control features from unstructured point clouds or monocular images.
    - **Mechanism**: A Transformer encoder is trained to map unstructured 3D head scans (or monocular images) to CUBE control feature lattices, enabling feed-forward scan registration and image-based reconstruction.
    - **Design Motivation**: The CUBE parameter space is compact (e.g., $8^3 \times 32 = 16\text{K}$ parameters), making it amenable to direct regression.

### Loss & Training

Vertex-to-vertex $\ell_2$ loss + normal consistency loss + Laplacian smoothing regularization. The encoder and CUBE decoder are trained end-to-end.

## Key Experimental Results

### Main Results

| Method | Type | Scan Registration Error↓ | Correspondence Accuracy↑ |
|--------|------|--------------------------|--------------------------|
| BPS | Basis Point Set | 2.85 | 82.3% |
| Shape-my-face | PointNet | 2.42 | 85.1% |
| ImFace | Implicit | 2.15 | 87.5% |
| **CUBE** | B-Spline | **1.89** | **91.2%** |

### Ablation Study

| Configuration | Scan Error↓ | Notes |
|---------------|-------------|-------|
| Full CUBE | 1.89 | High-dim features + MLP residual |
| w/o MLP residual | 2.35 | B-spline basis only |
| 3D control points (conventional) | 2.78 | No high-dim features |
| Lattice $16^3$ | 1.85 | More control points |
| Lattice $4^3$ | 2.45 | Fewer control points |

### Key Findings

- The MLP residual contributes significantly (removal increases error by 24%), underscoring the importance of high-frequency detail modeling.
- High-dimensional control features vs. 3D control points: error decreases from 2.78 to 2.35 (−15%), demonstrating enhanced expressiveness.
- An $8^3$ lattice is already sufficient: scaling to $16^3$ yields only marginal improvement.

## Highlights & Insights

- Adapting NURBS — a classical CAD representation — to face modeling and augmenting it with learnable features constitutes an elegant hybrid design.
- Preservation of the local support property enables interactive editing: local face regions can be manipulated by swapping or modifying individual control features.
- The two-stage decoding strategy (coarse B-spline + fine MLP) is generalizable to other geometric representations.

## Limitations & Future Work

- The model is face-specific; hair and accessories are not modeled.
- Fine detail under extreme expressions may be inferior to implicit representations.
- The choice of lattice resolution requires balancing expressiveness and efficiency.
- The approach is extendable to other body parts such as the full body or hands.

## Related Work & Insights

- **vs. 3DMM (FLAME)**: 3DMMs rely on linear PCA bases, whereas CUBE employs B-spline volumes combined with an MLP, achieving greater expressiveness while retaining local controllability.
- **vs. ImFace**: ImFace is an implicit SDF representation that requires Marching Cubes for mesh extraction; CUBE directly outputs a mesh via template-based query.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The hybrid combination of B-spline volumes and high-dimensional features is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on two applications: scan registration and image-based reconstruction.
- **Writing Quality**: ⭐⭐⭐⭐ The representation design is described with clarity.
- **Value**: ⭐⭐⭐⭐ Practically valuable for editable face modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models](../../ICCV2025/3d_vision/representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](../../ICCV2025/3d_vision/sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](../../ICCV2025/3d_vision/billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[CVPR 2026\] 3D-IDE: 3D Implicit Depth Emergent](3d-ide_3d_implicit_depth_emergent.md)
- [\[CVPR 2026\] ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion](actionmesh_animated_3d_mesh_generation_with_temporal_3d_diffusion.md)

</div>

<!-- RELATED:END -->
