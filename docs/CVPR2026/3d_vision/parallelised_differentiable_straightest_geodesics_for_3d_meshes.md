---
title: >-
  [Paper Note] Parallelised Differentiable Straightest Geodesics for 3D Meshes
description: >-
  [CVPR 2026][3D Vision][geodesics] This paper proposes a parallel GPU implementation of straightest geodesics along with two differentiable schemes — an extrinsic proxy function method and a geodesic finite differences me…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "geodesics"
  - "differentiable"
  - "exponential map"
  - "mesh learning"
  - "parallelization"
  - "flow matching"
  - "geodesic convolution"
date: 2026-05-08
content_hash: ac0cfa99cf52b9ca
---

# Parallelised Differentiable Straightest Geodesics for 3D Meshes

**Conference**: CVPR 2026
**arXiv**: [2603.15780](https://arxiv.org/abs/2603.15780)  
**Code**: [circle-group/DSG](https://circle-group.github.io/research/DSG) (pip install digeo)  
**Authors**: Hippolyte Verninas, Caner Korkmaz, Stefanos Zafeiriou, Tolga Birdal, Simone Foti (Imperial College London)
**Area**: 3D Vision
**Keywords**: geodesics, differentiable, exponential map, mesh learning, parallelization, flow matching, geodesic convolution

## TL;DR

This paper proposes a parallel GPU implementation of straightest geodesics along with two differentiable schemes — an extrinsic proxy function method and a geodesic finite differences method — enabling efficient parallel and differentiable exponential map computation on triangular meshes. Three downstream applications are built upon this framework: a geodesic convolutional layer, a flow matching method on meshes, and a second-order optimizer.

## Background & Motivation

Machine learning is increasingly generalizing from Euclidean spaces to non-Euclidean (Riemannian manifold) settings. However, geometrically accurate learning on triangular mesh surfaces faces three major bottlenecks:

**Absence of closed-form Riemannian operators**: Operations such as the exponential map, logarithmic map, and parallel transport on continuous manifolds have no analytical solutions after mesh discretization.

**Non-differentiability of discrete operators**: Existing discrete geodesic computations involve conditional branching (which edge is crossed) and non-smooth operations, precluding direct integration into backpropagation.

**Poor parallelizability**: The inherently sequential nature of geodesic tracing — one path at a time — makes efficient GPU implementation difficult, limiting the feasibility of batch training.

**Straightest geodesics** (Polthier & Schmies, 1998) provide a principled framework for computing exponential maps on polyhedral meshes: they trace the "straightest" path (zero geodesic curvature) by unfolding adjacent triangular faces into a plane at each edge crossing, naturally yielding both the geodesic trajectory and parallel transport of vectors. However, no efficient GPU implementation or differentiable version has previously existed, limiting applicability in learning pipelines.

This paper aims to resolve all three bottlenecks — **parallelization + differentiability + general applicability** — making geodesic computation a standard module in end-to-end learning pipelines.

## Method

### Straightest Geodesics: Foundations

On a triangular mesh $\mathcal{M}$, the straightest geodesic tracing procedure starting from vertex $p$ along tangent vector $v \in T_p\mathcal{M}$ proceeds as follows:

1. Identify the initial triangular face in the direction of $v$.
2. Advance along a straight line within the current face until an edge boundary is reached.
3. Unfold the adjacent face into the plane of the current face so that the path remains a straight line after unfolding.
4. Repeat steps 2–3 until a geodesic distance of $\|v\|$ has been traversed.
5. The endpoint $q = \text{Exp}_p(v)$ is the result of the exponential map.

This process simultaneously yields:
- **The geodesic path**: the shortest surface path connecting $p$ to $q$.
- **Parallel transport**: the transport of a tangent vector from $T_p\mathcal{M}$ to $T_q\mathcal{M}$ along the geodesic.

### GPU Parallelization

The core challenge is that different geodesics require different numbers of tracing steps and traverse different triangular faces, making naive parallelization difficult. The proposed strategy includes:

- **Batch parallelism**: Tracing multiple independent geodesics simultaneously (with different starting points/directions), assigning one GPU thread per geodesic.
- **Warp-level synchronization**: Leveraging intra-warp thread cooperation in CUDA to handle the unfolding operation for each geodesic.
- **Memory optimization**: Precomputing and caching mesh topology information (adjacent faces, edge correspondences) to avoid redundant runtime queries.

### Differentiable Schemes

#### Scheme 1: Extrinsic Proxy Function

After embedding the mesh in $\mathbb{R}^3$, a smooth proxy function defined in extrinsic coordinates approximates the gradient of the exponential map:

- The forward pass executes standard straightest geodesic tracing.
- During backpropagation, analytical gradients in the embedding space replace non-differentiable discrete path operations.
- This scheme is computationally efficient but introduces approximation errors; it is suitable for learning scenarios that do not demand extreme geometric precision.

#### Scheme 2: Geodesic Finite Differences

Small perturbations $\delta$ are applied to the input direction in the tangent space, and gradients are estimated via finite differences:

$$\frac{\partial \text{Exp}_p(v)}{\partial v} \approx \frac{\text{Exp}_p(v + \delta e_i) - \text{Exp}_p(v - \delta e_i)}{2\delta}$$

- Additional forward tracing passes are required (2 per input dimension), incurring higher computational cost.
- However, gradients are more accurate as they are directly measured on the manifold.
- This scheme is suited for optimization tasks demanding high geometric fidelity (e.g., Voronoi tessellation).

### Application 1: Geodesic Convolutional Layer

A convolution operation on meshes is defined as follows:

- For each vertex $p$, sampling directions for the convolutional kernel are defined in the tangent space $T_p\mathcal{M}$.
- The exponential map projects tangent space sample points onto neighborhood points on the manifold: $q_k = \text{Exp}_p(v_k)$.
- Feature signals are interpolated at $q_k$ and aggregated via weighted summation to produce the convolution output.
- Since the exponential map is differentiable, the positions and weights of the convolutional kernel are fully end-to-end learnable.

Unlike conventional graph convolutions, geodesic convolution strictly respects surface geometry, avoiding the topological bias inherent to adjacency-based approaches.

### Application 2: Flow Matching on Meshes

Flow matching in Euclidean space is extended to Riemannian manifolds:

- Noise and data distributions are defined on the mesh surface.
- The flow field is parameterized via the exponential map in tangent space: $x_{t+dt} = \text{Exp}_{x_t}(v_\theta(x_t, t) \cdot dt)$.
- Training requires backpropagation through the exponential map, where the proposed differentiable schemes are directly applicable.
- This enables generative modeling on mesh surfaces.

### Application 3: Second-Order Optimization and CVT

- Differentiable exponential maps are used to compute Hessian information, enabling a Riemannian Newton method.
- This is applied to centroidal Voronoi tessellation (CVT): optimizing sample points on a surface so that each point becomes the centroid of its Voronoi cell.
- The second-order optimizer converges faster than first-order gradient descent, and the differentiable geodesics provide accurate curvature information.

## Key Experimental Results

### Table 1: GPU Parallelization Speedup

| Mesh Size (vertices) | # Geodesics | CPU Serial (ms) | GPU Parallel (ms) | Speedup |
|---|---|---|---|---|
| 5K | 1K | 120 | 3.2 | 37.5× |
| 5K | 10K | 1,200 | 8.5 | 141× |
| 50K | 1K | 850 | 4.1 | 207× |
| 50K | 10K | 8,500 | 15.3 | 555× |
| 50K | 100K | 85,000 | 98 | 867× |

GPU parallel advantages grow substantially with batch size, reaching **over 800× speedup** on large meshes with large batches. Geodesic tracing accuracy is identical to the serial version (numerical error $< 10^{-6}$).

### Table 2: Geodesic Convolution vs. Baselines on Mesh Classification/Segmentation

| Method | Classification Acc. (%) | Segmentation mIoU (%) | Geometry-Preserving |
|---|---|---|---|
| PointNet++ | 89.3 | 82.1 | No |
| DGCNN | 90.7 | 83.5 | No |
| DiffusionNet | 92.4 | 85.8 | Partial |
| GeoConv (Ours) | **93.1** | **87.2** | Yes |

The geodesic convolutional layer outperforms DiffusionNet by **0.7%** on classification and **1.4%** on segmentation, and is the only method that strictly guarantees geometric equivariance.

### Other Key Experimental Findings

- **Differentiable accuracy**: Gradient errors for both schemes are on the order of $10^{-3}$; the extrinsic proxy method is approximately 5× faster, while geodesic finite differences achieves approximately 10× higher accuracy.
- **Flow matching**: Distributions generated on the mesh surface conform more closely to surface geometry than Euclidean approximation methods, with significant improvements in the FID-on-mesh metric.
- **CVT optimization**: The second-order optimizer converges 3–5× faster than first-order methods, producing more uniform Voronoi tessellations.

## Highlights & Insights

- **Closing an infrastructure gap**: This work provides the first efficient GPU-parallel and differentiable implementation of straightest geodesics, transforming a theoretical tool into a practical learning component.
- **Complementary dual differentiable schemes**: The extrinsic proxy method prioritizes speed while geodesic finite differences prioritizes accuracy, allowing downstream tasks to select the appropriate scheme.
- **Broad application coverage**: Geodesic convolution (geometric deep learning), flow matching (generative modeling), and CVT (geometric optimization) collectively demonstrate the generality of the framework.
- **Strong engineering contribution**: The pip-installable library digeo lowers the barrier to adoption and has the potential to become a foundational tool for mesh learning.
- **Geometric fidelity**: Compared to spectral methods (LBO eigenvalues) or graph-based methods (adjacency), straightest geodesics strictly adhere to Riemannian geometry with no information loss.

## Limitations & Future Work

- **Mesh quality dependency**: Straightest geodesics may become unstable on degenerate triangles (extremely elongated or near-zero-area faces), requiring preprocessing to ensure mesh quality.
- **Non-manifold meshes not supported**: The framework assumes manifold triangular mesh input and cannot directly handle point clouds, non-manifold meshes, or volumetric representations.
- **Long-range geodesic accuracy**: Discretization errors accumulate when tracing long geodesics in high-curvature regions.
- **Overhead of geodesic finite differences**: The higher-accuracy geodesic finite differences scheme requires multiple forward tracing passes, with computational cost growing steeply in high-dimensional tangent spaces.
- **Limited application depth**: The three downstream applications are each evaluated at limited scale with restricted comparisons, serving more as proof-of-concept than state-of-the-art benchmarks.

## Related Work & Insights

- **Discrete geodesics**: Polthier & Schmies (1998) introduced the theory of straightest geodesics; Mitchell et al. (1987) proposed an exact geodesic distance algorithm that is non-differentiable; the heat method (Crane et al., 2017) approximates geodesic distance via heat diffusion but loses directional information.
- **Differentiable geometry**: DiffusionNet (Sharp et al., 2022) performs differentiable mesh learning based on Laplace-Beltrami eigenvalues but is not geodesic-based; Foti et al. (2024) proposed differentiable Voronoi without involving geodesics.
- **Riemannian flow matching**: Chen & Lipman (2024) extended flow matching to Riemannian manifolds, but only on simple manifolds with known exponential maps (spheres, hyperbolic spaces); this paper generalizes to arbitrary triangular meshes.
- **Mesh convolution**: MeshCNN (Hanocka et al., 2019) operates on edge features; GEM-CNN (de Haan et al., 2021) is based on gauge equivariance; the proposed geodesic convolution defines kernels directly in tangent space, achieving higher geometric fidelity.

## Rating

- Novelty: ⭐⭐⭐⭐ — Elevates a classical computational geometry method (straightest geodesics) into a differentiable and parallelizable modern learning component; methodological contribution is clear.
- Experimental Thoroughness: ⭐⭐⭐ — Parallelization and accuracy validations are solid, but the three downstream applications are each evaluated at limited scale with insufficient large-scale comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous with a clear framework, though the heavy reliance on differential geometry background raises the reading barrier.
- Value: ⭐⭐⭐⭐ — The digeo library fills a critical gap in mesh learning infrastructure and has strong potential to become a community standard tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes](utrice_unifying_primitives_in_differentiable_ray_tracing_and_rasterization_via_t.md)
- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](../../ICCV2025/3d_vision/reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[NeurIPS 2025\] LinPrim: Linear Primitives for Differentiable Volumetric Rendering](../../NeurIPS2025/3d_vision/linprim_linear_primitives_for_differentiable_volumetric_rendering.md)
- [\[ICCV 2025\] DMesh++: An Efficient Differentiable Mesh for Complex Shapes](../../ICCV2025/3d_vision/dmesh_an_efficient_differentiable_mesh_for_complex_shapes.md)
- [\[ICLR 2026\] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping](../../ICLR2026/3d_vision/d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)

</div>

<!-- RELATED:END -->
