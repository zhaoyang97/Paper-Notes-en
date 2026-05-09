---
title: >-
  [Paper Note] Geometry Distributions
description: >-
  [ICCV 2025][3D Vision][Geometry representation] This paper proposes Geometry Distributions (GeomDist), which models 3D geometry as a probability distribution over surface points and learns it via a diffusion model. Without assuming genus, connectivity, or boundary conditions, the method samples arbitrarily many surface points from Gaussian noise to represent geometry of arbitrary topology.
tags:
  - ICCV 2025
  - 3D Vision
  - Geometry representation
  - diffusion models
  - surface point distributions
  - neural compression
  - non-watertight meshes
date: 2026-05-08
content_hash: 789727c8c800aefd
---

# Geometry Distributions

**Conference**: ICCV 2025
**arXiv**: [2411.16076](https://arxiv.org/abs/2411.16076)
**Code**: Not released
**Area**: 3D Vision
**Keywords**: Geometry representation, diffusion models, surface point distributions, neural compression, non-watertight meshes

## TL;DR

This paper proposes Geometry Distributions (GeomDist), which models 3D geometry as a probability distribution over surface points and learns it via a diffusion model. Without assuming genus, connectivity, or boundary conditions, the method samples arbitrarily many surface points from Gaussian noise to represent geometry of arbitrary topology.

## Background & Motivation

Existing 3D geometry representations each suffer from inherent limitations:
- **Meshes**: Inconsistent data structures, ill-suited for learning
- **Voxels**: Memory-intensive, demanding high resolution
- **Point clouds**: Finite sampling, lacking connectivity information
- **SDFs**: Unable to represent thin structures and non-watertight geometry

The core insight is that any surface can be approximated by a sufficiently large set of sampled points, and generative models can theoretically sample unlimited data from a distribution. Accordingly, geometry is modeled as a distribution $\Phi_{\mathcal{M}}$ over surface points such that $\mathbf{x} \sim \Phi_{\mathcal{M}} \Rightarrow \mathbf{x} \in \mathcal{M}$.

## Method

### Problem Formulation

Given a surface $\mathcal{M} \subset \mathbb{R}^3$, the goal is to learn a mapping $\mathcal{E}$ from a Gaussian distribution to the surface point distribution. This mapping is learned via a diffusion model $D_\theta(\cdot, \cdot)$ satisfying the ODE:

$$\mathrm{d}\mathbf{x} = \frac{\mathbf{x} - D_\theta(\mathbf{x}, t)}{t} \mathrm{d}t$$

### Forward Sampling (Gaussian → Surface)

Starting from Gaussian noise $\mathbf{x}_0 = T\mathbf{n}$, points are iteratively updated as:

$$\mathbf{x}_{i+1} = \mathbf{x}_i + (t_{i+1} - t_i) \cdot \frac{\mathbf{x}_i - D_\theta(\mathbf{x}_i, t_i)}{t_i}$$

The endpoint $\mathbf{x}_N$ lies on the target surface. Sampling arbitrarily many Gaussian points allows the surface to be approximated to arbitrary precision.

### Inverse Sampling (Surface → Gaussian)

Starting from a surface point, the trajectory is traversed in reverse to map back to noise space:

$$\mathbf{x}_{i-1} = \mathbf{x}_i + (t_{i-1} - t_i) \cdot \frac{\mathbf{x}_i - D_\theta(\mathbf{x}_i, t_i)}{t_i}$$

This establishes a bijective correspondence between surface points and noise space.

### Training

A key design choice is **re-sampling** $2^{25}$ surface points at each epoch. After 1000 epochs, the network has observed sufficiently many surface points to simulate infinite sampling:

$$\arg\min_\theta \mathbb{E}_{\mathbf{x} \in \mathcal{M}} \mathbb{E}_{\mathbf{n} \sim \mathcal{N}} \mathbb{E}_{\sigma > 0} \|D_\theta(\mathbf{x} + \sigma\mathbf{n}, \sigma) - \mathbf{x}\|$$

### Network Architecture

Inspired by EDM, the network employs magnitude-preserving layer designs, consisting of 6 blocks with $C=512$ linear layers and a total of 5.53M parameters. Inputs and outputs are normalized to zero mean and unit variance.

## Key Experimental Results

### Comparison with SDF on Non-Watertight Objects

| Method | Parameters | Non-Watertight | Thin Structures |
|--------|-----------|----------------|-----------------|
| SDF (Instant-NGP) | 14M | ✗ | Poor |
| **GeomDist** | **5M** | **✓** | **Good** |

GeomDist represents open and non-watertight geometry that SDF cannot handle, using fewer parameters.

### Comparison with Vector Field Methods

| Method | Chamfer Distance (×10³) | Uniformity |
|--------|------------------------|------------|
| Vector field | 4.886 | Non-uniform |
| **GeomDist** | **3.218** | **Uniform** |

GeomDist outperforms vector field methods in both uniformity and geometric fidelity.

### Multi-Resolution Sampling

Sampling at varying resolutions from $n=2^{15}$ to $n=2^{19}$ on the Wukong mesh consistently yields accurate surface approximations, demonstrating continuous resolution adaptability.

## Highlights & Insights

1. **Universal representation**: No assumptions on genus, watertightness, or connectivity — a truly general geometric representation
2. **Infinite resolution**: Theoretically unlimited point sampling, unconstrained by fixed sampling density
3. **Compactness**: Complex geometry encoded in 5M parameters, far fewer than the 14M required by SDF
4. **Invertibility**: Forward and inverse sampling share the same trajectory, establishing a bijection between surface points and noise space
5. **Broad applicability**: Supports textured mesh representation, neural compression, dynamic modeling, and Gaussian splatting rendering

## Limitations & Future Work

- Training is time-consuming (several hours), unsuitable for real-time applications
- Each object requires training a separate network
- Surface extraction relies on post-processing (e.g., Ball Pivoting for connectivity)
- Scalability to large-scale scenes has not been demonstrated

## Related Work & Insights

- SDF/UDF: Coordinate-based neural representations
- Point-E, NeuralPoints: Point cloud generation
- EDM: Diffusion model framework

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (A fundamentally new perspective: geometry as distribution)
- Technical Depth: ⭐⭐⭐⭐⭐ (ODE framework + network design + training strategy)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple object categories + ablations + applications)
- Value: ⭐⭐⭐⭐ (General representation with broad potential)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Neural Compression for 3D Geometry Sets](neural_compression_for_3d_geometry_sets.md)
- [\[NeurIPS 2025\] GOATex: Geometry & Occlusion-Aware Texturing](../../NeurIPS2025/3d_vision/goatex_geometry_occlusion-aware_texturing.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[ICCV 2025\] G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection](g2sf_geometry-guided_score_fusion_for_multimodal_industrial_anomaly_detection.md)
- [\[ICCV 2025\] SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)

<!-- RELATED:END -->
