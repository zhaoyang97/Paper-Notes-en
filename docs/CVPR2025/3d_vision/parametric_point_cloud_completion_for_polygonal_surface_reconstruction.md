---
title: >-
  [Paper Note] Parametric Point Cloud Completion for Polygonal Surface Reconstruction
description: >-
  [CVPR 2025][3D Vision][Point Cloud Completion] This work proposes PaCo, a new paradigm for parametric point cloud completion. Instead of predicting individual points, PaCo infers parametric planar primitives from incomplete point clouds. Through hierarchical encoding, proxy generation, and bipartite matching optimization, it directly bridges the gap between incomplete data and high-quality polygonal surface reconstruction.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Parametric Primitives"
  - "Polygonal Surface Reconstruction"
  - "Bipartite Matching"
  - "Sequence Generation"
date: 2026-05-08
content_hash: 6aea158aef6a500c
---

# Parametric Point Cloud Completion for Polygonal Surface Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2503.08363](https://arxiv.org/abs/2503.08363)  
**Code**: [Project Page](https://parametric-completion.github.io)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Completion, Parametric Primitives, Polygonal Surface Reconstruction, Bipartite Matching, Sequence Generation

## TL;DR

This work proposes PaCo, a new paradigm for parametric point cloud completion. Instead of predicting individual points, PaCo infers parametric planar primitives from incomplete point clouds. Through hierarchical encoding, proxy generation, and bipartite matching optimization, it directly bridges the gap between incomplete data and high-quality polygonal surface reconstruction.

## Background & Motivation

Polygonal surface reconstruction aims for compact planar representations, but existing methods heavily rely on the integrity of the input:

1. **Gap between Point Cloud Completion and Surface Reconstruction**: Existing point cloud completion methods (such as PoinTr) recover individual points but ignore piecewise planar geometric structures, making them unsuitable for direct use in polygonal reconstruction.
2. **Dependency of Primitive Assembly Methods**: Methods like PolyFit/KSR/COMPOD require high-quality planar primitives (usually extracted by RANSAC), yielding poor results on incomplete inputs.
3. **Limitations of Traditional Completion**: Additionally recovered points may not lie on the plane (non-coplanar), which degrades planar structures.

This paper proposes a new paradigm of "parametric completion": instead of recovering individual points, it infers parametric primitives (plane parameters + inlier distributions).

## Method

### Overall Architecture

Given an incomplete point cloud $X$, PaCo first hierarchically encodes it into plane proxies $V$. Then, a proxy generator generates proxy proposals of the complete surface. The parameter recovery module extracts plane parameters and inlier distributions from the proxies. Finally, a primitive selector filters valid primitives. During training, bipartite matching is utilized to establish correspondences between predicted and GT primitives.

### Key Designs

**1. Hierarchical Encoding: Points $\to$ Patches $\to$ Planes**

- **Function**: Aggregates incomplete point clouds hierarchically into plane proxies containing structural information.
- **Mechanism**: Points are first grouped into plane segments $S$ using GoCoPP. Point patch features $X'$ are obtained via PoinTr encoding. A mapping $f'$ from points to patches and then to planes is established through a lookup table, followed by sum pooling to aggregate plane proxies, with normal embeddings injected: $v_i = \text{sum}(X_i') + \Phi(n_i)$
- **Design Motivation**: Progressively aggregating low-level point features into high-level plane features preserves local geometric details while capturing plane-level structural information.

**2. Parameter Recovery Module (Parameter Estimation + Point Distribution + Primitive Selection)**

- **Function**: Restores complete parametric primitives from proxies, including plane parameters, inlier distribution, and confidence.
- **Mechanism**: The parameter estimator represents plane parameters in polar coordinates $(r_i, \theta_i, \varphi_i)$ to avoid axis-aligned degradation. The point distributor predicts the polar angles $(\theta_{ij}, \varphi_{ij})$ of each inlier, with the radius derived from plane parameters: $r_{ij} = \frac{r_i}{\cos(\Delta\varphi)\sin\theta_{ij}\sin\theta_i + \cos\theta_{ij}\cos\theta_i}$. The primitive selector outputs a confidence score $\kappa_i \in [0,1]$.
- **Design Motivation**: Polar coordinates prevent degradation issues associated with Cartesian representations. Deriving the radius from angles and plane parameters ensures points lie strictly on the plane. Variable-length primitives accommodate surfaces of varying complexity.

**3. Bipartite Matching Optimization Framework**

- **Function**: Establishes optimal correspondences between predicted and GT primitives to address order permutation issues in set prediction.
- **Mechanism**: The Hungarian algorithm is used to minimize the total matching cost: $\hat{\sigma} = \arg\min_{\sigma \in \Pi} \sum_i^M C(p_i, \hat{p}_{\sigma(i)})$. The cost includes a semantic term (classification) and geometric terms (normals + plane Chamfer + repulsion loss). GT is padded with $\emptyset$ to have the same cardinality as predictions.
- **Design Motivation**: Inspired by DETR, bipartite matching naturally handles unordered sets of primitives of varying sizes, bypassing the limitation of predefined primitive counts.

### Loss & Training

$$\mathcal{L}_{total} = \sum_{i=1}^{M} (\mathcal{L}_{cls}^{(i)} + \beta_1 \mathcal{L}_{norm}^{(i)} + \beta_2 \mathcal{L}_{cp}^{(i)} + \beta_3 \mathcal{L}_{rep}^{(i)}) + \beta_4 \mathcal{L}_{co}$$

Here, $\mathcal{L}_{cls}$ is the binary cross-entropy (BCE) loss for primitive classification, $\mathcal{L}_{norm}$ represents the normal cosine + L2 loss, $\mathcal{L}_{cp}$ denotes the per-primitive Chamfer distance, $\mathcal{L}_{rep}$ is the repulsion loss (promoting uniform distribution), and $\mathcal{L}_{co}$ represents the global Chamfer distance.

## Key Experimental Results

### Main Results

| Completion Method + PolyFit | CD↓(×100) | HD↓(×100) | NC↑ | FR↓(%) |
|------------------|-----------|-----------|-----|--------|
| PCN | 14.10 | 20.73 | 0.620 | 71.27 |
| FoldingNet | 12.07 | 21.24 | 0.814 | 3.54 |
| PoinTr | 10.57 | 16.43 | 0.822 | 25.92 |
| AdaPoinTr | 9.86 | 15.36 | 0.831 | 23.24 |
| **PaCo (Ours)** | **2.23** | **7.44** | **0.936** | **0.25** |

### Comparison Under Different Occlusion Levels

| Occlusion Level | PaCo CD↓ | AdaPoinTr CD↓ | Gain |
|---------|---------|-------------|------|
| Simple (25%) | 1.45 | 6.82 | **4.7×** |
| Moderate (50%) | 2.15 | 9.42 | **4.4×** |
| Hard (75%) | 3.09 | 13.34 | **4.3×** |

### Key Findings

- PaCo improves CD by 4–5 times compared to the best baseline, AdaPoinTr, and reduces the failure rate from 23.24% to 0.25%.
- It maintains excellent performance under a 75% missing rate (hard condition), demonstrating that parametric completion is highly effective for severely incomplete data.
- The recovered primitives yield the best results when directly fed into three different reconstructors (PolyFit, COMPOD, and KSR).

## Highlights & Insights

1. **Paradigm Shift**: Shifting the focus from "recovering points" to "recovering primitives" represents a major breakthrough in point cloud completion.
2. **Structure Preservation**: The parametric representation ensures that reconstructed points lie strictly on planes, eliminating non-coplanar noise.
3. **DETR-style Set Prediction**: Bipartite matching elegantly solves the problem of predicting an unordered set of varying numbers of primitives.

## Limitations & Future Work

- Only handles planar primitives; curved surfaces are not supported.
- Relies on the quality of the prior segmentation from GoCoPP.
- The ABC dataset consists mainly of CAD models; evaluation on real scanned scenes is still required.
- Future work can extend this approach to parametric curved surfaces (e.g., cylinders, spheres).

## Related Work & Insights

- **PoinTr/AdaPoinTr**: SOTAs in point cloud completion, but they do not preserve planar structures.
- **PolyFit/COMPOD**: Polygonal reconstruction methods that rely on complete inputs.
- **DETR**: The source of inspiration for set prediction using bipartite matching.

## Rating

⭐⭐⭐⭐⭐ — Exceptional paradigm shift that directly bridges the gap between point cloud completion and surface reconstruction. The method is rigorously designed, thoroughly ablated, and exhibits impressive robustness under extremely incomplete data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)
- [\[CVPR 2025\] GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors](genpc_zero-shot_point_cloud_completion_via_3d_generative_priors.md)
- [\[CVPR 2025\] ESCAPE: Equivariant Shape Completion via Anchor Point Encoding](escape_equivariant_shape_completion_via_anchor_point_encoding.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](../../ICCV2025/3d_vision/revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)
- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](../../ICCV2025/3d_vision/constraint-aware_feature_learning_for_parametric_point_cloud.md)

</div>

<!-- RELATED:END -->
