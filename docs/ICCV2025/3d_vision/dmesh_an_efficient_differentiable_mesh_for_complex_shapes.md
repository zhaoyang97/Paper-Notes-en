---
title: >-
  [Paper Note] DMesh++: An Efficient Differentiable Mesh for Complex Shapes
description: >-
  [ICCV 2025][3D Vision][Differentiable mesh] This paper proposes DMesh++, which replaces weighted Delaunay triangulation (WDT) with a Minimum-Ball algorithm as the tessellation function for differentiable meshes…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Differentiable mesh"
  - "triangulation"
  - "point cloud reconstruction"
  - "multi-view reconstruction"
  - "Minimum-Ball algorithm"
date: 2026-05-08
content_hash: f27cd077bf6cf171
---

# DMesh++: An Efficient Differentiable Mesh for Complex Shapes

**Conference**: ICCV 2025
**arXiv**: [2412.16776](https://arxiv.org/abs/2412.16776)  
**Code**: Available on project page  
**Area**: 3D Vision
**Keywords**: Differentiable mesh, triangulation, point cloud reconstruction, multi-view reconstruction, Minimum-Ball algorithm

## TL;DR
This paper proposes DMesh++, which replaces weighted Delaunay triangulation (WDT) with a Minimum-Ball algorithm as the tessellation function for differentiable meshes, reducing computational complexity from $O(N)$ to $O(\log N)$. The method achieves up to 32× speedup on complex shapes while preserving desirable properties such as no self-intersections and few degenerate triangles.

## Background & Motivation

Triangle meshes are the most widely used shape representation in downstream tasks (rendering, simulation, etc.) due to their efficiency, flexibility, and controllability. However, mesh connectivity is inherently discrete, and the number of possible connectivity configurations grows exponentially with the number of vertices, making it challenging to use meshes as a differentiable shape representation.

Existing approaches and their limitations:
- **Neural implicit representations** (e.g., SDF/UDF + Marching Cubes): Efficient but generally unable to handle open surfaces and non-orientable geometry.
- **Data-driven methods** (e.g., MeshGPT, SpaceMesh): Transformer-based connectivity prediction, but not robust to outliers and self-intersections.
- **DMesh**: A probabilistic method proposed by Son et al. that assigns an existence probability to each face via WDT, avoiding self-intersections and outliers. However, WDT has $O(N)$ complexity and is difficult to parallelize on GPUs, requiring ~800 ms for $N=100\text{K}$ points in 3D.

**Key Challenge**: DMesh's probabilistic framework is theoretically elegant, but the computational bottleneck of WDT severely limits its applicability to complex shapes.

**Core Idea**: Design a Minimum-Ball condition to replace WDT, preserving DMesh's advantages of no self-intersections and few degenerate triangles while reducing complexity to $O(\log N)$ for efficient GPU parallelization.

## Method

### Overall Architecture

DMesh++ follows DMesh's probabilistic paradigm: each point is represented by a $(d+1)$-dimensional vector ($d$-dimensional position + a real-valued scalar $\psi$), eliminating the need for WDT weights $w$. The existence probability of a face $F$ is defined as $\Lambda(F) = \Lambda_{min}(F) \times \Lambda_{real}(F)$, where $\Lambda_{min}$ determines whether $F$ satisfies the Minimum-Ball condition and $\Lambda_{real}$ determines whether its vertices lie on the shape surface.

Reconstruction follows a multi-stage optimization: (1) initialize point features; (2) optimize point positions; (3) optimize real values; (4) subdivide the mesh to add detail; (5) iterate the above steps.

### Key Designs

1. **Minimum-Ball Condition (Definition 3.1)**:

    - *Function*: Define the minimum enclosing ball $B_F$ of face $F$ (the smallest-radius ball passing through all vertices of $F$). If the interior of $B_F$ contains no other points from $\mathbb{P}$, then $F \in \mathbb{F}_{min}$.
    - *Mechanism*: Validity of a face is reduced to a nearest-neighbor search. Given the center $B_F^c$ and radius $B_F^r$ of $B_F$, find the nearest neighbor of $B_F^c$ in $\mathbb{P} - F$:
$$d(B_F, \mathbb{P}) = \min_{p \in \mathbb{P}-F} \|p - B_F^c\| - B_F^r$$
$$F \in \mathbb{F}_{min} \Leftrightarrow d(B_F, \mathbb{P}) > 0$$
   - *Design Motivation*: Nearest-neighbor search is highly parallelizable on GPUs (via KD-tree), whereas WDT is nearly impossible to parallelize due to inherent race conditions.

2. **Differentiable Probability Computation**:

    - *Function*: Soften the discrete Minimum-Ball condition into a continuous probability using a sigmoid function.
    - *Mechanism*:
$$\Lambda_{min}(F) = \sigma\!\left(d(B_F, \mathbb{P}) \cdot \alpha_{min}\right)$$
     where $\alpha_{min}$ is a sharpness constant. The center and radius of $B_F$ are computed in a differentiable manner by solving geometric equations.
   - *Design Motivation*: Gradient propagation is maintained so that mesh topology can dynamically change during optimization of point positions.

3. **Theoretical Guarantees**:

    - $\mathbb{F}_{min} \subseteq \mathbb{F}_{dt}$ (the Minimum-Ball face set is a subset of the Delaunay triangulation face set), thus inheriting the no-self-intersection property.
    - Although $\mathbb{F}_{min}$ does not necessarily tessellate the entire convex hull, this is beneficial for shape reconstruction as it avoids forcing "imaginary" faces into regions where none should exist.
    - Minimum-Ball inherits Delaunay triangulation's property of minimizing degenerate (thin) triangles.

4. **Reconstruction Pipeline Optimizations**:

    - Multi-stage optimization: positions and real values are optimized separately to avoid instability from joint optimization.
    - Mesh subdivision: new points are inserted on existing faces to add geometric detail.
    - Loss functions: Chamfer Distance for point cloud reconstruction; differentiable rendering loss for multi-view reconstruction.
    - Periodic caching of nearest neighbors to accelerate the optimization process.
    - All experiments are conducted on AMD EPYC 7R32 CPU + NVIDIA A10 GPU.

### Loss & Training

- Point cloud reconstruction: primarily minimizes Chamfer Distance loss.
- Multi-view reconstruction: differentiable rendering loss + step-wise per-stage optimization (positions → real values → colors).
- Periodic nearest-neighbor caching for acceleration.

## Key Experimental Results

### Main Results (3D Point Cloud Reconstruction, 50 Hand-Selected Models)

| Method | CD(×10⁻³)↓ | F1↑ | NC↑ | AR↓ | SI↓ | #Vertices | #Faces | Time (s) |
|--------|-----------|-----|-----|-----|-----|-----------|--------|----------|
| VoroMesh | 19.591 | 0.352 | 0.855 | 145.7 | 0 | 64561 | 129338 | 11 |
| PSR | 10.164 | 0.392 | 0.943 | 5.218 | 0 | 139857 | 279739 | 4 |
| PoNQ | 1.578 | 0.402 | 0.934 | 2.288 | 0 | 47254 | 94664 | 32 |
| DMesh | 0.154 | 0.289 | 0.921 | 1.961 | 0 | 5815 | 13088 | 1147 |
| **DMesh++** | **0.033** | **0.480** | **0.938** | **1.814** | 0 | 25396 | 55546 | **282** |

### Ablation Study (Tessellation Efficiency Comparison)

| Configuration | 2D Speed | 3D Speed | GPU Memory |
|---------------|----------|----------|------------|
| DMesh (WDT) | Baseline | Baseline | Baseline |
| DMesh++ (Minimum-Ball) | **16× faster** | **32× faster** | **75–96% reduction** |
| $N=200\text{K}$, 3D | DMesh: ~800 ms | DMesh++: **168 ms** | Significantly lower |

### Key Findings

- DMesh++ improves CD over DMesh by approximately 5× (0.033 vs. 0.154) while reducing reconstruction time by ~4× (282 s vs. 1147 s).
- On 500 randomly sampled Thingi10K models, DMesh++ achieves significantly lower CD on both closed and open surfaces compared to all baselines.
- GPU memory usage is drastically reduced: 96% in 2D and 75% in 3D.
- DMesh fails with out-of-memory errors on complex 2D drawings, while DMesh++ reconstructs them successfully.
- Compared to implicit-representation-based methods (PSR, PoNQ), DMesh++ achieves superior CD and F1 scores and handles open surfaces.

## Highlights & Insights

1. **Elegance of the algorithmic substitution**: The Minimum-Ball condition is a mathematically elegant replacement for WDT that preserves all theoretical guarantees (no self-intersections, few degenerate triangles) while reducing the problem to a highly parallelizable nearest-neighbor search.
2. **Progressive topology changes**: During continuous optimization of point features, discrete mesh topology naturally evolves (as illustrated in the vase reconstruction in Figure 3). This continuous–discrete coupling is particularly appealing.
3. **General 2D/3D applicability**: The same framework handles both 2D line-segment meshes and 3D triangle meshes, demonstrating the theoretical generality of the approach.
4. **Practical value**: As a foundational differentiable mesh component in ML pipelines, DMesh++ can be directly adopted by generative models (e.g., MeshGPT variants).

## Limitations & Future Work

- $\mathbb{F}_{min}$ is a strict subset of $\mathbb{F}_{dt}$, so the resulting mesh may contain "holes" (some Delaunay faces are excluded from the Minimum-Ball set).
- Multi-view reconstruction quality still lags behind implicit methods such as NeuS.
- Reconstruction time (282 s/model) remains slow for practical deployment.
- The method requires point cloud or multi-view image input; single-image or text-conditioned generation is not supported.
- Integration of DMesh++ with large-scale generative models has not yet been validated.

## Related Work & Insights

- DMesh [Son et al., NeurIPS 2024] is the direct predecessor; this work resolves its core computational bottleneck.
- FlexiCubes [Shen et al.] and DMTet [Shen et al.] are alternative approaches based on differentiable isosurface extraction but cannot handle open surfaces.
- Key insight: when using mesh representations in machine learning, the critical challenge is not making the mesh itself differentiable, but making **the conditions that determine mesh existence** differentiable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Minimum-Ball condition is a precise and elegant replacement for WDT with a clear theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on 500+ models; however, multi-view reconstruction comparisons include relatively few baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured with rigorous logical flow from theory to implementation to experiments.
- Value: ⭐⭐⭐⭐ Resolves the core bottleneck of DMesh, making differentiable meshes practical for complex shapes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] χ: Symmetry Understanding of 3D Shapes via Chirality Disentanglement](kh_symmetry_understanding_of_3d_shapes_via_chirality_disentanglement.md)
- [\[ICCV 2025\] Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models](representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)

</div>

<!-- RELATED:END -->
