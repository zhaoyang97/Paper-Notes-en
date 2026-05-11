---
title: >-
  [Paper Note] UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes
description: >-
  [CVPR 2026][3D Vision][Differentiable Ray Tracing] UTrice proposes replacing Gaussian ellipsoids with triangles as unified primitives for differentiable ray tracing…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Differentiable Ray Tracing"
  - "Triangle Primitives"
  - "3D Gaussian Splatting"
  - "Novel View Synthesis"
  - "BVH Acceleration"
date: 2026-05-08
content_hash: 78251adb82333bbf
---

# UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes

**Conference**: CVPR 2026
**arXiv**: [2512.04421](https://arxiv.org/abs/2512.04421)
**Code**: [https://github.com/waizui/UTrice](https://github.com/waizui/UTrice)
**Area**: 3D Vision
**Keywords**: Differentiable Ray Tracing, Triangle Primitives, 3D Gaussian Splatting, Novel View Synthesis, BVH Acceleration

## TL;DR

UTrice proposes replacing Gaussian ellipsoids with triangles as unified primitives for differentiable ray tracing, enabling direct triangle traversal within an OptiX BVH without any proxy geometry. The method significantly outperforms 3DGRT in rendering quality while maintaining real-time performance, and is natively compatible with triangles optimized by the rasterization-based Triangle Splatting, thereby achieving primitive unification across rasterization and ray tracing pipelines.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the dominant approach for novel view synthesis due to its superior rendering quality and real-time performance. Subsequent works such as 2DGS replace 3D Gaussians with 2D planar Gaussian disks, and Triangle Splatting (3DTS) further replaces Gaussians with triangles, achieving continuous improvements in fidelity and training speed. Meanwhile, ray tracing—a classical technique in computer graphics—enables physically realistic effects such as depth of field, refraction, and environment illumination. 3DGRT pioneered the integration of ray tracing into the 3DGS framework.

**Limitations of Prior Work**: The fundamental issue with 3DGRT is that Gaussian kernels are defined over infinitely smooth convex supports and cannot serve directly as BVH geometric primitives. Consequently, 3DGRT constructs a regular icosahedron as a proxy geometry for each Gaussian particle to bound it, and then performs ray–intersection tests on this proxy. This introduces substantial overhead: proxy geometry construction consumes memory, BVH build time accounts for a significant fraction of total runtime, and custom intersection programs increase implementation complexity.

**Key Challenge**: Gaussian particles are inherently ill-suited as unified primitives for both ray tracing and rasterization. Rasterization pipelines can project, sort, and blend Gaussians, but ray tracing requires precise geometric intersections within a BVH. The unbounded nature of Gaussians necessitates proxy geometry, resulting in two separate rendering pipelines with different primitive representations that cannot be unified.

**Goal**: (1) Eliminate the dependency on proxy geometry in ray tracing, reducing BVH construction and intersection overhead; (2) Enable both rasterization and ray tracing to share a single primitive type, allowing seamless interoperability between the two pipelines; (3) Maintain or improve rendering quality while preserving real-time performance.

**Key Insight**: Inspired by Triangle Splatting, the authors observe that triangles are the most universal primitives in computer graphics, natively supported by BVH acceleration structures and hardware ray tracing without any proxy geometry. Replacing Gaussian-plus-proxy with differentiable, optimizable triangle primitives for ray tracing addresses all of the above issues simultaneously.

**Core Idea**: Replace Gaussian primitives and their proxy geometry with differentiable triangles as the ray tracing primitive. Through a carefully designed window function and gradient propagation chain, triangles are optimized end-to-end within the ray tracing pipeline, while remaining natively compatible with rasterization-based pipelines.

## Method

### Overall Architecture

The UTrice pipeline proceeds as follows. Given an SfM point cloud, triangles are initialized following Triangle Splatting (sampling three vertices within a unit sphere). An index buffer is computed for these triangles, and the vertex array together with the index buffer are passed directly to OptiX to build a BVH. During ray tracing, each ray maintains a $k$-element buffer recording the $k$ nearest intersected triangles, iterating until a termination criterion is satisfied. The rendered output is compared with ground truth to compute a loss, and gradients are back-propagated to triangle parameters via custom CUDA kernels, with parameters updated using the Adam optimizer. The entire process requires no proxy geometry; triangles serve directly as the BVH primitives.

### Key Designs

#### 1. Differentiable Triangle Representation and Window Function

- **Function**: Defines the response function of a triangle at a ray intersection point, transforming hard-boundary geometry into a smoothly differentiable, optimizable primitive.
- **Mechanism**: Each triangle is parameterized by three vertices $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3 \in \mathbb{R}^3$, color $c$ (spherical harmonics, degree 3), smoothness factor $\sigma$, and opacity $o$. At the intersection point $\mathbf{p}$ of ray $r_o + t r_d$ with the triangle plane, the window function is defined as:

$$I(\mathbf{p}) = \text{ReLU}\left(\frac{\phi(\mathbf{p})}{\phi(\mathbf{s})}\right)^\sigma$$

where $\mathbf{s}$ is the incenter of the triangle, $\phi(\mathbf{p}) = \max_{i \in \{1,2,3\}} L_i(\mathbf{p})$, and $L_i(\mathbf{p}) = \mathbf{n}_i \cdot \mathbf{p} + d_i$ is the signed distance to edge $i$. This function evaluates to 1 at the incenter, 0 on the edges, and 0 outside. The smoothness factor $\sigma$ controls the interior response profile: as $\sigma \to 0$ the triangle approximates a fully filled solid; as $\sigma$ increases the response becomes more position-sensitive.

- **Design Motivation**: Unlike 3DTS, this window function is defined in world space rather than image space. The authors prove that this formulation is compatible with 3DTS—the window function $I$ is invariant to linear transformations under the same smoothness factor $\sigma$. Therefore, triangles optimized via 3DTS rasterization can be rendered directly with the proposed ray tracer without any additional processing. This property constitutes the mathematical foundation for primitive unification.

#### 2. GPU-Accelerated Ray Tracing (OptiX Integration)

- **Function**: Leverages the OptiX framework to implement hardware-accelerated triangle ray tracing.
- **Mechanism**: Because triangles are used as primitives, OptiX BVH construction requires only a vertex array and an index buffer as input—no custom primitives or bounding boxes are needed (as required by 3DGRT). Rays are launched in the Ray Generation program; an Any-hit program uses insertion sort to identify the $k$ nearest triangles along the ray direction, which are then composited in front-to-back order via alpha blending:

$$\mathcal{C} = \sum_{i=1}^{N} T_i \alpha_i c_i, \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

Traversal terminates when the accumulated transmittance falls below a threshold or all triangles have been processed.

- **Design Motivation**: Compared to 3DGRT's approach of constructing icosahedral proxies per Gaussian and performing custom intersection tests, using triangle primitives natively eliminates BVH construction overhead and custom bounding-box logic. Furthermore, accepting rays as origin-direction array inputs makes the system agnostic to camera model, naturally supporting non-pinhole configurations such as LiDAR and fisheye lenses.

#### 3. Gradient Propagation Chain for Triangle Vertices

- **Function**: Establishes a complete gradient path from rendering loss to triangle vertices, enabling the optimizer to rotate and scale triangles by adjusting vertex positions.
- **Mechanism**: Unlike Gaussians, triangles have no explicit position or scale parameters; all geometric properties are fully determined by three vertices. The gradient propagation chain is: loss $\to$ window function $I$ $\to$ edge normals $\mathbf{n}_i$ $\to$ vertices $\mathbf{v}_i$. The edge normal is computed as:

$$\mathbf{N}_i = [(\mathbf{v}_i - \mathbf{v}_{i+2}) \times (\mathbf{v}_{i+1} - \mathbf{v}_{i+2})] \times (\mathbf{v}_{i+1} - \mathbf{v}_i)$$

with unit edge normal $\mathbf{n}_i = \mathbf{N}_i / \|\mathbf{N}_i\|$. For $\sigma > 0$, different points within a triangle produce different responses; gradients of these responses propagate to the vertices, driving the optimizer to rotate and scale triangles to fit the ground truth.

- **Design Motivation**: The authors arrived at this stable and effective gradient propagation formula through extensive experimentation. The key insight is that cross products and normal normalization establish a differentiable link from the window function to vertex coordinates, allowing rendering loss to directly guide geometric deformation of triangles.

#### 4. Pruning and Densification Strategy (World-Space Occlusion Metric)

- **Function**: Determines which triangles should be removed and which should be subdivided.
- **Mechanism**: Pruning is based on three criteria: (a) triangles with opacity below a threshold are removed; (b) triangles with $\omega = T \cdot o \cdot \rho$ below a threshold are removed ($T$: transmittance, $o$: opacity, $\rho$: window function response); (c) triangles hit by fewer than two camera views are removed. For densification, since optimization operates in world space rather than image space, the image-space footprint metric of 3DTS cannot be directly applied. The authors propose a world-space occlusion metric: the angle between the vector from each vertex to the ray origin and the vector from the triangle centroid to the ray origin is measured. This metric inherently accounts for distance—a small triangle close to the camera subtends an equivalent solid angle to a large triangle far away.
- **Design Motivation**: View-based pruning prevents degenerate triangles with vanishingly small gradients from producing NaNs; the world-space occlusion metric enables MCMC densification to correctly distinguish large from small triangles in world space—without it, training speed degrades by up to 5× or fails to converge entirely.

### Loss & Training

The total loss function is:

$$\mathcal{L} = (1 - \lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{\text{D-SSIM}} + \lambda_o \mathcal{L}_o + \lambda_n \mathcal{L}_n + \lambda_s \mathcal{L}_s$$

where $\mathcal{L}_1$ and $\mathcal{L}_{\text{D-SSIM}}$ are pixel-level and structural similarity losses, respectively; $\mathcal{L}_n$ is the normal loss (from 2DGS); $\mathcal{L}_o$ is the opacity loss; and $\mathcal{L}_s$ is a size loss that encourages larger triangle areas: $\mathcal{L}_s = 2 \cdot \|(\mathbf{v}_1 - \mathbf{v}_0) \times (\mathbf{v}_2 - \mathbf{v}_0)\|_2^{-1}$. Training uses PyTorch with custom CUDA kernels and the Adam optimizer; densification is performed every 500 iterations from iteration 500 to 25,000.

## Key Experimental Results

### Main Results

Evaluation on the Mip-NeRF 360 and Tanks & Temples datasets, compared against 3DGS, 2DGS, 3DTS, and 3DGRT:

| Method | Mip-NeRF360 PSNR↑ | SSIM↑ | LPIPS↓ | T&T PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|----------|-------|--------|----------|-------|--------|
| 3DGS | 28.69 | 0.870 | 0.182 | 23.14 | 0.841 | 0.183 |
| 2DGS | 28.56 | 0.862 | 0.190 | 23.13 | 0.832 | 0.212 |
| 3DTS | 28.95 | 0.876 | 0.153 | 23.06 | 0.842 | 0.164 |
| 3DGRT | 28.32 | 0.859 | 0.235 | 22.76 | 0.844 | 0.201 |
| **UTrice** | **28.70** | **0.866** | **0.163** | **22.88** | **0.849** | **0.150** |

UTrice improves LPIPS over 3DGRT by approximately **30%** on Mip-NeRF 360 (0.235→0.163) and **25%** on T&T (0.201→0.150), demonstrating substantially superior perceptual quality and detail preservation. Rendering speed:

| Method | Mip-NeRF360 FPS↑ | T&T FPS↑ |
|--------|----------|----------|
| 3DGRT (performance) | 78 | 190 |
| 3DGRT (quality) | 55 | 143 |
| UTrice | 37 | 119 |

UTrice is approximately 30% slower than 3DGRT (quality), but the pipeline has not yet been optimized and remains within the near-real-time regime.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Notes |
|---------------|-------|-------|--------|-------|
| Full model | 28.70 | 0.866 | 0.163 | Complete model |
| w/o World-space occlusion metric | N/A | N/A | N/A | 5× training slowdown on *bicycle*, fails to converge |
| w/o View-based pruning | N/A | N/A | N/A | NaN encountered on *stump*, training collapses |
| w/o $\mathcal{L}_n$ | 28.69 | 0.865 | 0.163 | Slight quality degradation |
| w/o $\mathcal{L}_s$ | 28.54 | 0.864 | 0.164 | Quality degradation; triangle count increases by 0.1% |

### Key Findings

- **The world-space occlusion metric is essential**: Without it, MCMC densification cannot distinguish large from small triangles, directly causing training collapse. This is the most critical adaptation when migrating from image-space rasterization to world-space ray tracing.
- **View-based pruning prevents numerical instability**: Vanishingly small gradients from degenerate triangles underflow to NaN through repeated multiplication; view-based pruning mitigates this by removing triangles hit by only a single view.
- **3DGRT over-smooths high-frequency regions**: The smooth Gaussian kernel causes loss of fine detail and even introduces high-frequency color noise in distant regions (e.g., the glass area of the *truck* scene). UTrice does not exhibit these artifacts.
- **UTrice's primitive count is comparable to 3DTS** (Mip-NeRF 360 average: 3.32M vs. 3.22M) and substantially fewer than 3DGRT (3.36M), with a more pronounced advantage on T&T (2.19M vs. 3.88M).

## Highlights & Insights

- **Primitive unification is the core contribution**: Because the window function is invariant to linear transformations in world space, triangles optimized via 3DTS rasterization can be rendered directly using UTrice's ray tracer. This enables a two-stage workflow—fast rasterization-based training followed by a switch to ray tracing for depth-of-field and refraction effects—with seamless transition between stages.
- **Eliminating proxy geometry is an elegant solution**: Much of 3DGRT's complexity stems from icosahedral proxies and custom intersection programs. By changing the primitive type, UTrice eliminates these issues entirely, reducing BVH construction to a native OptiX triangle pipeline.
- **The world-space occlusion metric is a transferable technique**: Any method performing primitive optimization in world space (rather than via image-space projection) can adopt angle-based rather than pixel-area-based size measurement, which inherently accounts for distance.
- **Generality of the ray input interface**: The ray tracer accepts only ray origin and direction arrays, independent of any camera model, enabling straightforward extension to panoramic, fisheye, LiDAR, and other non-pinhole imaging systems.

## Limitations & Future Work

- **High primitive count**: The resulting triangle soup lacks mesh connectivity; adjacent vertices are stored redundantly, increasing memory and computational overhead. A shared-vertex mesh structure could reduce this redundancy.
- **Training speed is approximately 2× slower than 3DGRT**: The pipeline contains computational redundancies and lacks handling mechanisms for degenerate triangles (extremely small or large).
- **PSNR does not surpass 3DGS**: Planar primitives (triangles, 2D Gaussians) generally underperform 3D Gaussians on PSNR, as the smooth Gaussian kernel artificially inflates PSNR—an artifact of over-smoothing rather than a genuine quality advantage.
- **Rendering speed has room for improvement**: The current implementation is unoptimized (37 FPS vs. 55 FPS); engineering optimization is expected to narrow or close the gap with 3DGRT.
- **Single-bounce only**: Current refraction and reflection effects are limited to single-bounce ray tracing; a full dielectric BSDF model has not been implemented, limiting physical accuracy.

## Related Work & Insights

- **vs. 3DGRT**: 3DGRT uses Gaussians with icosahedral proxies for ray tracing, requiring custom BVH primitives and intersection programs; UTrice uses triangles directly, leveraging native OptiX support for simpler and more efficient BVH construction. UTrice leads by a large margin on LPIPS (~30%) while being moderately slower in FPS (37 vs. 55).
- **vs. Triangle Splatting (3DTS)**: 3DTS uses triangles for rasterization; UTrice uses the same triangles for ray tracing. Both pipelines share identical primitives. UTrice achieves comparable perceptual quality to 3DTS while additionally enabling depth-of-field, refraction, and other ray-tracing effects.
- **vs. 2DGS**: Both methods use planar primitives, but 2DGS employs 2D Gaussian disks whereas UTrice uses triangles. Triangles are more general and better preserve high-frequency detail and sharp edges.
- The "unified primitive" paradigm introduced in this paper offers broader inspiration for any 3D representation required to simultaneously support multiple rendering pipelines.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of replacing Gaussians with triangles as ray tracing primitives is natural yet effective; the core contributions lie in the differentiable triangle gradient design and world-space adaptation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons on standard benchmarks; ablation studies clearly establish the necessity of each component; validation on additional scenes and downstream applications is limited.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, complete mathematical derivations (including supplementary material), and effective visual aids.
- Value: ⭐⭐⭐⭐ — Achieves primitive unification across rasterization and ray tracing pipelines, providing a foundational framework for simultaneously exploiting both rendering paradigms; practical value is high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)
- [\[CVPR 2026\] FreeScale: Scaling 3D Scenes via Certainty-Aware Free-View Generation](freescale_scaling_3d_scenes.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](../../ICCV2025/3d_vision/radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[NeurIPS 2025\] LinPrim: Linear Primitives for Differentiable Volumetric Rendering](../../NeurIPS2025/3d_vision/linprim_linear_primitives_for_differentiable_volumetric_rendering.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
