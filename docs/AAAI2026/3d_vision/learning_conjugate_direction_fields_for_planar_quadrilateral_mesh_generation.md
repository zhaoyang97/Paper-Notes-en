---
title: >-
  [Paper Note] Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation
description: >-
  [AAAI 2026][3D Vision][Planar Quadrilateral Mesh] This paper proposes an efficient, data-driven method based on DGCNN to generate conjugate direction fields (CDFs), bypassing the high computational overhead of traditional non-linear optimization. It supports user-stroke-guided controllable CDF generation, speeding up CDF computation by 1 to 2 orders of magnitude. Along with the method, a large-scale dataset containing over 50,000 free-form surfaces is released.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Planar Quadrilateral Mesh"
  - "Conjugate Direction Fields"
  - "Deep Learning"
  - "Architectural Design"
  - "Controllable Generation"
date: 2026-05-08
content_hash: 00ee1ca1f575a441
---

# Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation

**Conference**: AAAI 2026  
**arXiv**: [2511.11865](https://arxiv.org/abs/2511.11865)  
**Code**: [https://github.com/jiongtj/Learning-CDF](https://github.com/jiongtj/Learning-CDF)  
**Area**: 3D Vision  
**Keywords**: Planar Quadrilateral Mesh, Conjugate Direction Fields, Deep Learning, Architectural Design, Controllable Generation

## TL;DR

This paper proposes an efficient, data-driven method based on DGCNN to generate conjugate direction fields (CDFs), bypassing the high computational overhead of traditional non-linear optimization. It supports user-stroke-guided controllable CDF generation, speeding up CDF computation by 1 to 2 orders of magnitude. Along with the method, a large-scale dataset containing over 50,000 free-form surfaces is released.

## Background & Motivation

**Planar quadrilateral (PQ) meshes** are crucial in computer-aided design, particularly for the discretization of architectural surfaces. The key advantages of PQ meshes include: (1) face planarity significantly reduces the fabrication cost of physical materials such as glass; (2) compared to triangle meshes, the lower vertex degree reduces the complexity of support structures; (3) edge layouts are intuitive and aesthetically pleasing.

PQ mesh generation typically consists of two steps: first generating an initial quad mesh template, and then refining it through geometric optimization to make each face planar. The quality of the initial template relies on the **conjugate direction field (CDF)** on the surface—the conjugacy of the CDF ensures that the initial mesh faces are approximately planar, which is crucial for subsequent PQ mesh optimization.

**Key Challenge**: Unlike principal direction fields (PDFs) which are uniquely determined by surface geometry (except at umbilical points), CDFs are **not unique** and have high degrees of freedom. Users typically specify preferred directions via strokes, which act as constraints in non-linear optimization to compute the CDF. However, this non-linear optimization:
- **Is computationally expensive**: computational time increases sharply with mesh scale (taking ~17 seconds for ~20k faces, and ~40 seconds for ~60k faces).
- **Requires many iterations**: designers often need to repeatedly adjust strokes and recompute, leading to a poor interactive experience.
- **Hinders exploration**: makes real-time previewing of PQ mesh layouts corresponding to different CDF solutions impossible.

## Method

### Overall Architecture

Input: Triangle mesh $\mathcal{M} = \{\mathcal{V}, \mathcal{F}\}$ + user strokes $\mathcal{S} = \{\mathbf{S}_i\}$

Output: A pair of direction vectors $\{(\mathbf{u}_j, \mathbf{v}_j)\}$ on each triangular face, constituting the CDF

Pipeline: Feature extraction $\rightarrow$ CDF prediction $\rightarrow$ Global parameterization $\rightarrow$ Quad mesh extraction $\rightarrow$ Vertex perturbation optimization

### Key Designs

#### 1. **Feature Representation**

A 9-dimensional feature vector is constructed for each vertex by concatenating three parts:

- **Vertex position** $\mathbf{p}_i \in \mathbb{R}^3$: encodes mesh geometry
- **Vertex normal** $\mathbf{n}_i \in \mathbb{R}^3$: encodes local surface orientation
- **Stroke projection vector** $\mathbf{l}_i = \mathbf{p}_i^* - \mathbf{p}_i \in \mathbb{R}^3$: the vector from the vertex to the closest stroke point

The design of the stroke projection vector serves as a global stroke representation—it encodes the spatial relationship between each vertex and the stroke curves, rather than merely processing the strokes in isolation. Experiments demonstrate that this approach performs significantly better than using Point Cloud Transformer (PCT) to extract stroke features ($\delta$: 8.31° vs 20.98°).

#### 2. **Network Architecture**

**Feature Extraction Module**: Based on DGCNN, four EdgeConv layers are used to extract vertex features. Different from the original DGCNN, this work concatenates the local feature of each vertex with the global shape feature, which is then passed through a fully connected layer to obtain a 256-dimensional feature representation. DGCNN dynamically recomputes the local neighborhood at each layer, adaptively learning multi-scale geometric information.

**Prediction Module**: Two independent MLPs predict $\{\mathbf{u}_j\}$ and $\{\mathbf{v}_j\}$ respectively. The transformation from vertex features to face features is conducted via simple averaging. Each MLP consists of 3 layers (256$\rightarrow$128$\rightarrow$64$\rightarrow$3), with BatchNorm+ReLU in the first two layers, and the final layer directly outputting and normalizing to unit length.

#### 3. **Loss Function Design**

Five loss terms are carefully designed:

**Direction Alignment Loss $\mathcal{L}_d$**: measures the alignment between the predicted CDF and the ground truth. To handle sign ambiguity, the ground-truth vectors rotated by 90° are used, taking the minimum of the two possible correspondences:

$$\mathcal{L}_d = \frac{1}{m}\sum_{j=1}^{m} \min(E_j, E_j')$$

where $E_j = (\mathbf{u}_j \cdot \mathbf{u}_j^{*\perp})^2 + (\mathbf{v}_j \cdot \mathbf{v}_j^{*\perp})^2$

**Normal Consistency Loss $\mathcal{L}_{dn}$**: ensures the predicted directions are orthogonal to face normals:

$$\mathcal{L}_{dn} = \frac{1}{m}\sum_{j=1}^{m} (\mathbf{u}_j \cdot \mathbf{n}_j)^2 + (\mathbf{v}_j \cdot \mathbf{n}_j)^2$$

**Direction Smoothness Loss $\mathcal{L}_{ds}$**: ensures smooth transitions of the CDF between adjacent faces, reducing singularities:

$$\mathcal{L}_{ds} = \frac{1}{|\mathcal{N}|}\sum_{(j,k)\in\mathcal{N}} \min(E_{jk}, E_{jk}')$$

Parallel transport is utilized to handle the case of differing normals on adjacent faces.

**Stroke Consistency Loss $\mathcal{L}_{dc}$**: ensures the CDF directions align with the user strokes:

$$\mathcal{L}_{dc} = \frac{1}{|\mathcal{S}|}\sum_{\mathbf{S}_i \in \mathcal{S}} \frac{1}{|\mathcal{T}_i|}\sum_{k \in \mathcal{T}_i} D_k$$

**Field Regularization Loss $\mathcal{L}_{fr}$**: prevents predicting zero vectors:

$$\mathcal{L}_{fr} = \frac{1}{m}\sum_{j=1}^{m} (||\mathbf{u}_j||-1)^2 + (||\mathbf{v}_j||-1)^2$$

Total loss: $\mathcal{L}_{total} = \mathcal{L}_d + \lambda_1\mathcal{L}_{dn} + \lambda_2\mathcal{L}_{ds} + \lambda_3\mathcal{L}_{dc} + \lambda_4\mathcal{L}_{fr}$

All weights are set to 1.0.

### Loss & Training

- Dataset: B-spline surfaces consisting of 50,000 for training, 2,500 for validation, and 300 for testing.
- 2,601 sampled points and 5,000 faces per surface.
- Position and orientation normalized via PCA.
- Adam optimizer, learning rate of $1.0 \times 10^{-4}$, trained for 200 epochs.
- Hardware: Intel i9-14900K + NVIDIA RTX 4090.

## Key Experimental Results

### Main Results (Computational Efficiency Comparison)

CDF generation time comparison (vs. traditional optimization methods):

| Model | Face Count | Optimization Method | Ours | Gain |
|------|------|---------|---------|--------|
| Test Model 1 | 5,000 | 2.851s | 0.200s | 14.3× |
| Test Model 2 | 5,000 | 2.855s | 0.194s | 14.7× |
| Vase | 23,642 | 17.326s | 0.254s | 68.2× |
| Dome | 44,490 | 30.198s | 0.417s | 72.4× |
| Face | 60,077 | 40.412s | 0.571s | 70.8× |
| Garden (Arch) | 8,322 | 4.946s | 0.206s | 24.0× |
| Yas Island (Arch) | 7,029 | 3.766s | 0.204s | 18.5× |
| Aqua Dome (Arch) | 10,790 | 6.522s | 0.217s | 30.1× |

As the number of faces increases, the speedup becomes more significant: ~15× for 5k faces and ~71× for 60k faces. The computation time of the traditional method grows nearly linearly, whereas the learning-based method achieves almost flat growth.

### Ablation Study

Ablation study of loss functions on the test set (averaged over 300 models):

| Configuration | # Singularities | $\delta$ (Stroke Consistency) | $\theta$ (CDF Closeness) | Description |
|------|--------|--------------|-------------|------|
| Full model | 4.91 | 8.31° | 11.30° | Full model |
| w/o $\mathcal{L}_{ds}$ | 7.02 | 7.38° | 10.55° | Singularities increase significantly |
| w/o $\mathcal{L}_{dc}$ | 4.67 | 10.32° | 11.48° | Stroke consistency decreases |
| PCT stroke features | 8.97 | 20.98° | 19.06° | Significantly worse than our representation |

### Key Findings

- The smoothness loss $\mathcal{L}_{ds}$ is crucial for reducing singularities in PQ meshes (7.02 $\rightarrow$ 4.91).
- The stroke consistency loss $\mathcal{L}_{dc}$ reduces $\delta$ from 10.32° to 8.31°.
- The proposed stroke projection vector representation significantly outperforms PCT ($\delta$ is reduced by 60%, and $\theta$ is reduced by 41%), validating the importance of surface context awareness.
- The planarity of the PQ mesh (which is already excellent initially: $\eta_{\text{mean}} \approx 0.006$) is further improved to ~0.002 after vertex perturbation optimization.
- The method generalizes well to open-border surfaces, real architectural surfaces, and even closed models with varying topologies (such as the Stanford Bunny).
- Comparison with VectorHeat and NeurCross: VectorHeat cannot guarantee conjugacy, and NeurCross is limited to PDFs; neither is suitable for controllable CDF generation.

## Highlights & Insights

1. **Precise Problem Definition**: The non-uniqueness of CDFs is both a source of flexibility and a computational bottleneck. Bypassing non-linear optimization with a learning-based approach is a natural and effective strategy.
2. **Ingenious Stroke Representation**: The projection vectors encode the spatial relationship between each vertex and the strokes, implicitly propagating stroke information across the entire surface.
3. **Loss Function Handling Direction Ambiguity**: The inherent sign and correspondence  ambiguities of direction fields are elegantly resolved by employing 90° rotations and minimizing over correspondences.
4. **Large-scale Dataset Contribution**: The synthetic dataset of 50,000+ samples represents a substantial contribution. The training data incorporates actual design workflows by tracing streamlines from ground-truth CDFs to simulate user strokes.

## Limitations & Future Work

- The CDF cannot accurately align with sharp edges at sharp features (as the training data does not contain sharp features).
- There is no explicit control over the number and locations of singularities.
- The use of synthetic data generated solely from B-spline surfaces may limit generalization capability.
- Unsupervised methods to achieve better generalization have not yet been explored.
- The coverage and density of strokes affect the results, but a systematic analysis is currently lacking.

## Related Work & Insights

- Traditional methods (Liu et al. 2011) compute CDFs through constrained non-linear optimization, whereas this work entirely replaces it with learning.
- Sketch2PQ (Deng et al. 2022) predicts PQ meshes from 2D sketches, but is heavily restricted on 3D surfaces.
- VectorHeatNet learns vector fields but does not guarantee conjugacy.
- Insight: For other computational bottlenecks in architectural CAD (such as surface flattening and structural optimization), replacing optimization with learning may be equally applicable.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to address CDF generation using deep learning.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive efficiency comparison, complete ablation study, and generalization tests on both architectural and general 3D models.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear problem definition, rigorous mathematical derivation.
- Value: ⭐⭐⭐⭐ — Directly valuable for architectural design CAD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models](../../ICLR2026/3d_vision/quadgpt_native_quadrilateral_mesh_generation_with_autoregressive_models.md)
- [\[CVPR 2026\] Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation](../../CVPR2026/3d_vision/mesh-pro_asynchronous_advantage-guided_ranking_preference_optimization_for_artis.md)
- [\[AAAI 2026\] Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning](hierarchical_direction_perception_via_atomic_dot-product_operators_for_rotation-.md)
- [\[CVPR 2026\] Learning Convex Decomposition via Feature Fields](../../CVPR2026/3d_vision/learning_convex_decomposition_via_feature_fields.md)
- [\[AAAI 2026\] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)

</div>

<!-- RELATED:END -->
