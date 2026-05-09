---
title: >-
  [Paper Note] Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation
description: >-
  [AAAI 2026][3D Vision][Planar Quadrilateral Mesh] This paper proposes a data-driven approach based on DGCNN to efficiently generate conjugate direction fields (CDFs), bypassing the high computational cost of traditional nonlinear optimization. The method supports user stroke-guided controllable CDF generation, achieves a 1–2 order-of-magnitude speedup, and is accompanied by a large-scale dataset of 50,000+ free-form surfaces.
tags:
  - AAAI 2026
  - 3D Vision
  - Planar Quadrilateral Mesh
  - Conjugate Direction Field
  - Deep Learning
  - Architectural Design
  - Controllable Generation
date: 2026-05-08
content_hash: 7e24ee5d1b85110a
---

# Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation

**Conference**: AAAI 2026
**arXiv**: [2511.11865](https://arxiv.org/abs/2511.11865)
**Code**: [https://github.com/jiongtj/Learning-CDF](https://github.com/jiongtj/Learning-CDF)
**Area**: 3D Vision
**Keywords**: Planar Quadrilateral Mesh, Conjugate Direction Field, Deep Learning, Architectural Design, Controllable Generation

## TL;DR

This paper proposes a data-driven approach based on DGCNN to efficiently generate conjugate direction fields (CDFs), bypassing the high computational cost of traditional nonlinear optimization. The method supports user stroke-guided controllable CDF generation, achieves a 1–2 order-of-magnitude speedup, and is accompanied by a large-scale dataset of 50,000+ free-form surfaces.

## Background & Motivation

**Planar quadrilateral (PQ) meshes** are critical in computer-aided design, particularly for the discretization of architectural surfaces. Their key advantages include: (1) planarity of faces significantly reduces manufacturing costs for physical materials such as glass; (2) lower vertex valence compared to triangle meshes reduces structural complexity; and (3) edge layouts are visually intuitive and aesthetically appealing.

PQ mesh generation typically follows a two-stage pipeline: first generating an initial quadrilateral mesh layout, then refining it via geometric optimization to make each face approximately planar. The quality of the initial layout depends on the **conjugate direction field (CDF)** defined on the surface — conjugacy of the CDF ensures that the initial mesh faces are approximately planar, which is critical for subsequent PQ mesh optimization.

**Core Challenge**: Unlike principal curvature direction fields (PDFs), which are uniquely determined by surface geometry (except at umbilic points), CDFs are **non-unique** and possess high degrees of freedom. Users must specify preferred directions via strokes, which serve as constraints in a nonlinear optimization to compute the CDF. However, this nonlinear optimization is:
- **Computationally expensive**: cost grows sharply with mesh size (~17s for ~20k faces, ~40s for ~60k faces);
- **Iteratively demanding**: designers frequently need to adjust strokes and recompute, resulting in poor interactivity;
- **An obstacle to exploration**: real-time preview of PQ mesh layouts corresponding to different CDFs is infeasible.

## Method

### Overall Architecture

**Input**: Triangle mesh $\mathcal{M} = \{\mathcal{V}, \mathcal{F}\}$ + user strokes $\mathcal{S} = \{\mathbf{S}_i\}$

**Output**: Per-triangle-face direction vector pairs $\{(\mathbf{u}_j, \mathbf{v}_j)\}$ forming the CDF

**Pipeline**: Feature extraction → CDF prediction → Global parameterization → Quadrilateral mesh extraction → Vertex perturbation optimization

### Key Designs

#### 1. **Feature Representation**

A 9-dimensional feature vector is constructed for each vertex by concatenating three components:

- **Vertex position** $\mathbf{p}_i \in \mathbb{R}^3$: encodes mesh geometry
- **Vertex normal** $\mathbf{n}_i \in \mathbb{R}^3$: encodes local surface orientation
- **Stroke projection vector** $\mathbf{l}_i = \mathbf{p}_i^* - \mathbf{p}_i \in \mathbb{R}^3$: vector from the vertex to its nearest stroke point

The stroke projection vector serves as a global stroke representation — it encodes the spatial relationship between each vertex and the stroke curve, rather than processing the stroke in isolation. Experiments demonstrate that this representation substantially outperforms extracting stroke features via a Point Cloud Transformer (PCT) (δ: 8.31° vs. 20.98°).

#### 2. **Network Architecture**

**Feature Extraction Module**: Based on DGCNN, using 4 EdgeConv layers to extract per-vertex features. Unlike the original DGCNN, this work concatenates each vertex's local features with global shape features, then applies a fully connected layer to obtain a 256-dimensional feature representation. DGCNN dynamically recomputes local neighborhoods at each layer, enabling adaptive learning of multi-scale geometric information.

**Prediction Module**: Two independent MLPs predict $\{\mathbf{u}_j\}$ and $\{\mathbf{v}_j\}$ separately. Per-face features are obtained by averaging vertex features. Each MLP consists of 3 layers (256→128→64→3), with BatchNorm+ReLU applied to the first two layers; the final layer outputs predictions directly, which are then normalized to unit length.

#### 3. **Loss Function Design**

Five loss terms are carefully designed:

**Direction Alignment Loss $\mathcal{L}_d$**: Measures alignment between predicted CDFs and ground truth. The 90°-rotated ground truth vectors are used to handle sign ambiguity, and the minimum over two possible correspondences is taken:

$$\mathcal{L}_d = \frac{1}{m}\sum_{j=1}^{m} \min(E_j, E_j')$$

where $E_j = (\mathbf{u}_j \cdot \mathbf{u}_j^{*\perp})^2 + (\mathbf{v}_j \cdot \mathbf{v}_j^{*\perp})^2$

**Normal Consistency Loss $\mathcal{L}_{dn}$**: Ensures predicted directions are orthogonal to face normals:

$$\mathcal{L}_{dn} = \frac{1}{m}\sum_{j=1}^{m} (\mathbf{u}_j \cdot \mathbf{n}_j)^2 + (\mathbf{v}_j \cdot \mathbf{n}_j)^2$$

**Direction Smoothness Loss $\mathcal{L}_{ds}$**: Enforces smooth CDF transitions across adjacent faces to reduce singularities:

$$\mathcal{L}_{ds} = \frac{1}{|\mathcal{N}|}\sum_{(j,k)\in\mathcal{N}} \min(E_{jk}, E_{jk}')$$

Parallel transport is used to handle differing face normals between adjacent faces.

**Stroke Consistency Loss $\mathcal{L}_{dc}$**: Ensures CDF directions align with user strokes:

$$\mathcal{L}_{dc} = \frac{1}{|\mathcal{S}|}\sum_{\mathbf{S}_i \in \mathcal{S}} \frac{1}{|\mathcal{T}_i|}\sum_{k \in \mathcal{T}_i} D_k$$

**Field Regularization Loss $\mathcal{L}_{fr}$**: Prevents degenerate zero-vector predictions:

$$\mathcal{L}_{fr} = \frac{1}{m}\sum_{j=1}^{m} (||\mathbf{u}_j||-1)^2 + (||\mathbf{v}_j||-1)^2$$

Total loss: $\mathcal{L}_{total} = \mathcal{L}_d + \lambda_1\mathcal{L}_{dn} + \lambda_2\mathcal{L}_{ds} + \lambda_3\mathcal{L}_{dc} + \lambda_4\mathcal{L}_{fr}$

All weights are set to 1.0.

### Loss & Training

- Dataset: 50,000 training + 2,500 validation + 300 test B-spline surfaces
- Each surface has 2,601 sampled points and 5,000 faces
- Position and orientation normalized via PCA
- Adam optimizer, learning rate $1.0 \times 10^{-4}$, trained for 200 epochs
- Hardware: Intel i9-14900K + NVIDIA RTX 4090

## Key Experimental Results

### Main Results (Computational Efficiency Comparison)

CDF generation time comparison (vs. traditional optimization):

| Model | Faces | Optimization | Ours | Speedup |
|-------|-------|-------------|------|---------|
| Test Model 1 | 5,000 | 2.851s | 0.200s | 14.3× |
| Test Model 2 | 5,000 | 2.855s | 0.194s | 14.7× |
| Vase | 23,642 | 17.326s | 0.254s | 68.2× |
| Dome | 44,490 | 30.198s | 0.417s | 72.4× |
| Face | 60,077 | 40.412s | 0.571s | 70.8× |
| Garden (arch.) | 8,322 | 4.946s | 0.206s | 24.0× |
| Yas Island (arch.) | 7,029 | 3.766s | 0.204s | 18.5× |
| Aqua Dome (arch.) | 10,790 | 6.522s | 0.217s | 30.1× |

Speedup increases with mesh size: ~15× at 5k faces, ~71× at 60k faces. The traditional method scales nearly linearly, while the learning-based method exhibits nearly flat scaling.

### Ablation Study

Effect of ablating loss terms on the test set (averaged over 300 models):

| Configuration | Singularities | δ (stroke consistency) | θ (CDF proximity) | Notes |
|--------------|--------------|----------------------|------------------|-------|
| Full model | 4.91 | 8.31° | 11.30° | Complete model |
| w/o $\mathcal{L}_{ds}$ | 7.02 | 7.38° | 10.55° | Significant increase in singularities |
| w/o $\mathcal{L}_{dc}$ | 4.67 | 10.32° | 11.48° | Degraded stroke consistency |
| PCT stroke features | 8.97 | 20.98° | 19.06° | Far inferior to proposed representation |

### Key Findings

- Smoothness loss $\mathcal{L}_{ds}$ is critical for reducing PQ mesh singularities (7.02→4.91)
- Stroke consistency loss $\mathcal{L}_{dc}$ reduces δ from 10.32° to 8.31°
- The proposed stroke projection vector representation substantially outperforms PCT (60% reduction in δ, 41% reduction in θ), validating the importance of surface-context-aware encoding
- PQ mesh planarity is already favorable at initialization ($\eta_{\text{mean}} \approx 0.006$) and further improves to ~0.002 after vertex perturbation optimization
- The method generalizes to open-boundary surfaces, real architectural surfaces, and closed models with varied topology (e.g., Stanford Bunny)
- Compared with VectorHeat and NeurCross: VectorHeat cannot guarantee conjugacy; NeurCross is restricted to PDFs — neither is suitable for controllable CDF generation

## Highlights & Insights

1. **Precise problem formulation**: The non-uniqueness of CDFs is both a source of flexibility and a computational bottleneck; using a learning approach to bypass nonlinear optimization is natural and effective.
2. **Clever stroke representation design**: Projection vectors encode the spatial relationship between each vertex and the stroke, implicitly propagating stroke information across the entire surface.
3. **Elegant handling of direction ambiguity**: The 90°-rotation and minimum-over-correspondences strategy gracefully resolves the inherent sign and correspondence ambiguity of direction fields.
4. **Large-scale dataset contribution**: The synthetic dataset of 50,000+ samples is itself a practical contribution; training data construction simulates real design workflows by tracing streamlines from ground-truth CDFs to simulate user strokes.

## Limitations & Future Work

- CDFs cannot accurately align with sharp features, as training data does not include surfaces with such characteristics
- No explicit control over the number or placement of singularities
- Reliance on synthetic B-spline surface data may limit generalization
- Unsupervised approaches for improved generalization remain unexplored
- The influence of stroke coverage and density on results lacks systematic analysis

## Related Work & Insights

- Traditional methods (Liu et al. 2011) compute CDFs via constrained nonlinear optimization; this work replaces that process entirely with learning
- Sketch2PQ (Deng et al. 2022) predicts PQ meshes from 2D sketches but is limited in handling 3D surfaces
- VectorHeatNet learns vector fields but does not guarantee conjugacy
- Insight: Learning-as-optimization substitution may be equally applicable to other computational bottlenecks in architectural CAD, such as surface unfolding and structural optimization

## Rating

- Novelty: ⭐⭐⭐⭐ — First application of deep learning to CDF generation
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive efficiency comparisons, complete ablations, generalization tests on architectural and general 3D models
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem formulation and rigorous mathematical derivation
- Value: ⭐⭐⭐⭐ — Direct applicability to architectural design CAD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models](../../ICLR2026/3d_vision/quadgpt_native_quadrilateral_mesh_generation_with_autoregressive_models.md)
- [\[AAAI 2026\] Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning](hierarchical_direction_perception_via_atomic_dot-product_operators_for_rotation-.md)
- [\[AAAI 2026\] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](../../ICLR2026/3d_vision/topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](../../ICLR2026/3d_vision/learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)

</div>

<!-- RELATED:END -->
