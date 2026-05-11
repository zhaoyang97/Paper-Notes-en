---
title: >-
  [Paper Note] LinPrim: Linear Primitives for Differentiable Volumetric Rendering
description: >-
  [NeurIPS 2025][3D Vision][Novel View Synthesis] This paper proposes LinPrim, which replaces 3D Gaussian kernels with linear primitives (octahedra and tetrahedra) as the scene representation for novel view synthesis. Thro…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Differentiable Rendering"
  - "Volumetric Rendering"
  - "Polyhedral Primitives"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: df0219c13729eceb
---

# LinPrim: Linear Primitives for Differentiable Volumetric Rendering

**Conference**: NeurIPS 2025
**arXiv**: [2501.16312](https://arxiv.org/abs/2501.16312)
**Code**: [GitHub](https://nicolasvonluetzow.github.io/LinPrim/)
**Area**: 3D Vision
**Keywords**: Novel View Synthesis, Differentiable Rendering, Volumetric Rendering, Polyhedral Primitives, 3D Gaussian Splatting

## TL;DR

This paper proposes LinPrim, which replaces 3D Gaussian kernels with linear primitives (octahedra and tetrahedra) as the scene representation for novel view synthesis. Through a differentiable rasterization pipeline, LinPrim enables end-to-end optimization and achieves reconstruction quality comparable to 3DGS on real-world datasets using fewer primitives, while maintaining real-time rendering capability.

## Background & Motivation

Novel view synthesis (NVS) is a core task in 3D vision. Current state-of-the-art methods are primarily based on NeRF (implicit but computationally expensive) and 3D Gaussian Splatting (explicit and real-time). **Key Challenge**: 3DGS uses Gaussian kernels as primitives, which, despite their effectiveness, are unbounded and non-uniformly dense — properties that are less intuitive and difficult to integrate with conventional triangle-mesh-based 3D processing pipelines. The central research question is: can simple bounded polyhedral primitives achieve performance comparable to Gaussian kernels in NVS? The **Core Idea** is to use transparent octahedra/tetrahedra as scene primitives, exploiting the differentiability of ray–triangle intersection to enable gradient-based optimization, thereby exploring the design space of 3D scene representations.

## Method

### Overall Architecture

LinPrim builds upon the 3DGS pipeline, replacing the core scene representation and rendering process:

- **Input**: Multi-view RGB images + SfM point cloud
- **Scene Representation**: A collection of transparent octahedra/tetrahedra, each described by position, rotation, vertex distances, opacity, and spherical harmonic coefficients
- **Rendering**: Differentiable GPU rasterizer: preprocessing → global sorting + tile partitioning → per-pixel ray–triangle intersection → alpha compositing
- **Optimization**: L1 + SSIM loss, ADAM optimizer, gradients backpropagated to all primitive parameters

### Key Designs

1. **Octahedral Primitive Design**:

    - Vertices are constrained to lie along coordinate axes relative to the center; each antipodal vertex pair is described symmetrically by a single distance parameter
    - Each primitive uses 11 floats: 3 (position) + 4 (rotation quaternion) + 3 (three axis distances) + 1 (opacity) — identical in parameter count to a standard Gaussian kernel
    - 48 spherical harmonic coefficients are added to model view-dependent color
    - The enforced symmetry ensures the center coincides with the geometric centroid, simplifying bounding box computation

2. **Tetrahedral Primitive Design**:

    - Four evenly spaced basis vectors define directions from center to vertices; the corresponding distances are individually optimized
    - Due to the lack of symmetry, 12 floats are required (one more than the octahedron)
    - Performance is slightly inferior, but validates the generality of the framework

3. **Differentiable Rendering Pipeline**:

    - **Preprocessing**: Construct polyhedra → camera-space transformation → projective-space approximation → compute screen-space bounding boxes
    - **Rasterization**: Ray–triangle intersection via the Möller–Trumbore algorithm (MTIA), which is fully differentiable
    - **Opacity Computation**: A convex polyhedron yields at most two intersection points with a pixel ray; density is based on the distance between intersection points: $\sigma(\alpha) = -\frac{\log(1 - 0.99 \cdot \alpha)}{2 \cdot \min(d_x, d_y, d_z)}$
    - **Compositing**: Front-to-back alpha blending, terminated when accumulated opacity reaches 0.999

4. **Anti-Aliasing**:

    - 3D smoothing filter: enforces a minimum primitive size to ensure visibility in at least one pixel from at least one training view
    - 2D Mip-style filter: expands the projected bounding box to increase the minimum screen-space size of primitives (approximate method)

5. **Population Control**:

    - Initialized from SfM points with random rotations to ensure uniform coverage
    - Pruning: removes primitives that are overly transparent, too large, or occupy an excessive fraction of the screen
    - Cloning/splitting: applied to primitives with high positional gradients — small ones are cloned, large ones are split
    - Split sampling: new positions are sampled from a PDF using per-axis distances as standard deviations (feasible due to octahedral symmetry)
    - Compatible with the GS-MCMC population control strategy

### Loss & Training

The standard 3DGS loss is used: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{SSIM}$. Training runs for 30k iterations with the ADAM optimizer. Learning rates are adaptively scaled according to scene extent (maximum distance among training cameras). Consistent hyperparameters are used across all scenes.

## Key Experimental Results

### Main Results

| Method | ScanNet++ PSNR | ScanNet++ #Primitives | MipNeRF360 PSNR | MipNeRF360 #Primitives |
|--------|---------------|----------------------|-----------------|------------------------|
| 3DGS | 24.09 | 738k | 27.43 | 3.32M |
| Mip-Splatting | 24.12 | 977k | 27.79 | 4.17M |
| 3D Convex Splatting | 24.26 | 440k | 27.22 | 1.02M |
| **LinPrim (Octa)** | **24.04** | **255k** | 26.63 | **1.79M** |
| GS-MCMC (255k) | 24.50 | 255k | -- | -- |
| LinPrim+MCMC (255k) | **24.41** | 255k | -- | -- |
| LinPrim+MCMC (738k) | **24.55** | 738k | 27.04 | 3.32M |

### Ablation Study

| Configuration | ScanNet++ PSNR | MipNeRF360 PSNR | Notes |
|---------------|---------------|-----------------|-------|
| Octahedron | 24.04 / 0.849 SSIM | 26.63 / 0.803 SSIM | Default |
| Tetrahedron | 24.05 / 0.848 SSIM | 25.96 / 0.790 SSIM | On par on ScanNet++, worse on MipNeRF360 |
| LinPrim+MCMC vs GS-MCMC (equal count) | 24.41 vs 24.50 | -- | Small gap due to primitive type |
| Default vs MCMC population control | 24.04→24.41 | 26.63→27.04 | MCMC yields notable improvement |

### Key Findings

1. **Comparable quality with fewer primitives**: LinPrim approaches 3DGS quality on ScanNet++ using 255k primitives versus 3DGS's 738k, indicating higher representational efficiency for bounded polyhedra
2. **Octahedra outperform tetrahedra**: The superior symmetry leads to more stable optimization; the gap is pronounced on MipNeRF360 (26.63 vs. 25.96)
3. **Compatibility with MCMC population control**: Demonstrates that LinPrim is not tied to the 3DGS densification strategy and can benefit from more advanced alternatives
4. **Trade-offs of bounded primitives**: Produce sharper depth estimates in reflective/transparent regions, but exhibit blocky hard-boundary artifacts in under-observed areas
5. **Better suited for smaller scenes**: Densely covered, smaller scenes yield the best results, preserving geometric and visual clarity

## Highlights & Insights

- The contribution is more scientific than engineering-oriented: the work systematically investigates the viability of polyhedral primitives in NVS, expanding the design space for 3D scene representations
- The octahedron's 11-parameter description matches that of a Gaussian kernel, yet achieves better parameter efficiency (fewer primitives, comparable quality), suggesting that Gaussian kernels may be over-parameterized
- Ray–triangle intersection is inherently differentiable and well-established (MTIA), and in principle could directly leverage existing triangle-rendering hardware acceleration
- Applying random rotations after SfM initialization to ensure uniform coverage is a subtle but important design choice

## Limitations & Future Work

- PSNR on MipNeRF360 large scenes falls approximately 0.8 dB below 3DGS, a non-trivial gap
- Bounded primitives produce hard-boundary artifacts in sparsely observed regions, unlike the smooth fall-off of Gaussian kernels
- Current rendering throughput is lower than optimized Gaussian rasterizers, resulting in a practical FPS gap
- The population control strategy was originally designed for Gaussians; primitive-specific strategies could substantially improve performance
- Tetrahedra require special handling due to their asymmetry, and the current pipeline is not fully optimized for them

## Related Work & Insights

- Most closely related to 3D Convex Splatting (NeurIPS 2024), but LinPrim adopts simpler uniformly dense polyhedra rather than smooth convex bodies
- Relationship to EVER (uniformly dense ellipsoids): LinPrim further simplifies the primitive shape to a polyhedron
- Distinction from Beyond Gaussians (linear kernels): the latter retains elliptical, non-uniform density, whereas LinPrim uses bounded, uniform density
- Implication: the design space for NVS primitives extends far beyond Gaussian kernels; simple bounded, uniformly dense primitives merit further exploration

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic exploration of polyhedral primitives in NVS, though the core pipeline still draws heavily from 3DGS
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, two primitive types, MCMC comparisons, and reasonably complete ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous method derivation, and honest discussion of limitations
- Value: ⭐⭐⭐⭐ Expands the 3D representation design space, though performance has not yet surpassed 3DGS

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UTrice: Unifying Primitives in Differentiable Ray Tracing and Rasterization via Triangles for Particle-Based 3D Scenes](../../CVPR2026/3d_vision/utrice_unifying_primitives_in_differentiable_ray_tracing_and_rasterization_via_t.md)
- [\[NeurIPS 2025\] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)
- [\[NeurIPS 2025\] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](../../ICCV2025/3d_vision/radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[NeurIPS 2025\] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting](robust_neural_rendering_in_the_wild_with_asymmetric_dual_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
