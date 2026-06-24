---
title: >-
  [Paper Note] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Speedy-Splat is proposed to accelerate 3DGS rendering through two complementary pipelines: (1) SnugBox and AccuTile, which precisely locate the screen-space extent of Gaussians to reduce redundant pixel processing, and (2) efficient pruning (Soft and Hard Pruning) to reduce the number of Gaussians by over 90%. Combined, they achieve an average rendering speedup of 6.71×, alongside a 10.6× reduction in model size and a 1.4× trainin…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Rendering Acceleration"
  - "Pruning"
  - "Exact Tile Intersection"
  - "Real-time Rendering"
date: 2026-05-08
content_hash: 6c2f81a730b7c3e4
---

# Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives

**Conference**: CVPR 2025  
**arXiv**: [2412.00578](https://arxiv.org/abs/2412.00578)  
**Code**: [Project Page](https://speedysplat.github.io)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Rendering Acceleration, Pruning, Exact Tile Intersection, Real-time Rendering

## TL;DR

Speedy-Splat is proposed to accelerate 3DGS rendering through two complementary pipelines: (1) SnugBox and AccuTile, which precisely locate the screen-space extent of Gaussians to reduce redundant pixel processing, and (2) efficient pruning (Soft and Hard Pruning) to reduce the number of Gaussians by over 90%. Combined, they achieve an average rendering speedup of 6.71×, alongside a 10.6× reduction in model size and a 1.4× training speedup.

## Background & Motivation

- **Speed Bottleneck of 3DGS**: While 3DGS achieves real-time rendering on desktop GPUs, it is still far from real-time on edge devices (e.g., mobile phones), and suffers from large model sizes.
- **Two Factors in Rendering Cost**: The rendering cost of 3DGS is proportional to the number of Gaussians multiplied by the number of pixels processed per Gaussian. Speedy-Splat optimizes both factors simultaneously.
- **Conservativeness of Tile Assignment**: Original 3DGS uses the maximum eigenvalue to calculate a bounding circle and then an axis-aligned bounding square to determine the tile intersection region, heavily overestimating the actual span of Gaussians by ignoring opacity $\sigma_i$ and elliptical shape information.
- **Gaussian Redundancy**: Numerous studies show that about 90% of Gaussians in 3DGS models are redundant and can be pruned without sacrificing visual quality.
- **Practical Limitations of Existing Pruning**: PUP 3D-GS proposes a principled Hessian-based pruning method, which yields high performance but requires $N \times 36$ storage, making it impossible to integrate into the training loop.

## Method

### Overall Architecture

Speedy-Splat consists of two parts: exact tile intersection (SnugBox + AccuTile, which is plug-and-play and does not alter rendering results) and efficient pruning (Soft Pruning integrated into the densification phase + Hard Pruning applied after densification). Both are integrated into the 3DGS training pipeline.

### Key Designs

**Design 1: SnugBox — Exact Axis-Aligned Bounding Box**
- **Function**: Computes tight bounding boxes for each Gaussian by considering opacity, replacing the conservative bounding circle estimation of original 3DGS.
- **Mechanism**: The actual pixel extent of Gaussian $\mathcal{G}_i$ is determined by the threshold $\alpha_i \geq \frac{1}{255}$. Combining $\sigma_i$ and the 2D covariance $\Sigma_{i_{2D}}$ yields the ellipse equation $t = ax_d^2 + 2bx_dy_d + cy_d^2$, where $t = 2\log(255\sigma_i)$. Taking the extremum $\partial y_d / \partial x_d = 0$ yields $y_{\min}, y_{\max}$. Swapping $x, y$ and $a, c$ yields $x_{\min}, x_{\max}$, forming a tight bounding box.
- **Design Motivation**: The bounding square in the original method severely overestimates the extent when the Gaussian has low opacity or high anisotropy. The overhead of SnugBox is constant, requiring only one invocation each in preprocess and duplicateWithKeys.

**Design 2: AccuTile — Exact Tile-Ellipse Intersection**
- **Function**: Further identifies tiles that truly intersect with the ellipse from within the rectangular region defined by SnugBox.
- **Mechanism**: Processing line/column-wise along the shorter dimension of the SnugBox rectangle. Each iteration only requires computing the two intersection points between the ellipse and a boundary line (exploiting shared boundaries of adjacent lines), keeping the time complexity proportional to the length of the shorter side of the rectangle. Key insight: The extremal points of SnugBox are the unique inflection points; since the ellipse is monotonic on non-inflection rows/columns, calculating the boundary intersections is sufficient to determine the minimum/maximum tile range.
- **Design Motivation**: The rectangle of SnugBox still contains corner tiles that do not intersect with the ellipse. AccuTile further eliminates these redundant allocations with minimal computational overhead.

**Design 3: Efficient Soft/Hard Pruning — Integrated Pruning Training**
- **Function**: Progressively prunes redundant Gaussians during the training process, reducing the number of Gaussians by over 90%.
- **Mechanism**: Reparameterizes the Hessian from PUP 3D-GS to compute the sensitivity $\tilde{U}_i = \log|\nabla_{g_i} I_\mathcal{G} \nabla_{g_i} I_\mathcal{G}^T|$ for the 2D projected values $g_i(p)$ of each Gaussian. This reduces the storage requirement from $N \times 36$ to $N \times 1$ (a 36× compression) and allows direct utilization of pixel-parallel gradients from the rendering kernel. Soft Pruning removes 80% of low-sensitivity Gaussians at fixed intervals during the densification phase, while Hard Pruning further prunes 30% after densification.
- **Design Motivation**: The original PUP method requires pixel-level gradients with respect to 3D parameters $\mu, s$ (disrupting 3DGS's efficient gradient flow). Reparameterizing to the gradients of 2D projected values seamlessly fits the render-backpropagation architecture of 3DGS.

### Loss & Training

Identical to standard 3DGS: $L = \|I_\mathcal{G}(\phi) - I_{\text{gt}}\|_1 + L_{\text{D-SSIM}}(I_\mathcal{G}(\phi), I_{\text{gt}})$. Pruning does not alter the loss function and is executed periodically during training.

## Key Experimental Results

### Main Results: Acceleration Across Rendering Pipeline Stages (Average)

| Configuration | Preprocess | Radix Sort | Render | Overall | Speedup |
|------|-----------|-----------|--------|---------|-------|
| Baseline | 0.665ms | 1.551ms | 4.483ms | 7.478ms | 1× |
| +SnugBox | 0.656ms | 0.729ms | 2.344ms | 4.102ms | 1.82× |
| +AccuTile | 0.668ms | 0.612ms | 2.062ms | 3.748ms | 2.00× |
| +Soft Pruning | 0.370ms | 0.404ms | 1.337ms | 2.381ms | 3.14× |
| **+Hard Pruning** | **0.091ms** | **0.215ms** | **0.619ms** | **1.114ms** | **6.71×** |

### Tanks & Temples Truck Scene Example

| Metric | Original 3DGS | Speedy-Splat |
|------|---------|-------------|
| Number of Gaussians | ~100% | <10% (Reduced by >90%) |
| PSNR Drop | - | Only minor drop |
| Rendering Speedup | 1× | 6.2× |
| Training Speedup | 1× | 1.38× |

### Key Findings

1. SnugBox + AccuTile, acting as plug-and-play modules, achieve a 2× speedup without modifying any rendering results (zero accuracy loss).
2. When accumulating all optimizations, an average rendering speedup of 6.71×, a 10.6× model size reduction, and a 1.4× training speedup are achieved.
3. Reparameterization of the efficient pruning score compresses storage from $N \times 36$ to $N \times 1$, enabling real-time pruning during training.
4. The two-stage (Soft + Hard) pruning, applied respectively before and after densification, is more effective than single post-processing pruning.

## Highlights & Insights

- **Accuracy is Efficiency**: SnugBox/AccuTile reduce redundant computation through more precise geometric calculations rather than approximate simplification—accuracy actually brings speed.
- **36× Storage Compression via Hessian Reparameterization**: Elegantly translates the sensitivity of 3D parameters to the sensitivity of 2D projected values, perfectly fitting the 3DGS architecture.
- **Outstanding Engineering Innovation**: Analyzes bottlenecks by diving deep into each part of the CUDA rendering pipeline, providing a detailed function-by-function timing analysis.
- All methods are orthogonal to existing 3DGS variants and can be combined built upon them.

## Limitations & Future Work

- AccuTile's row/column-wise traversal might introduce extra overhead for extremely large Gaussians (which cover many tiles).
- Currently validated only on static scenes; applicability to dynamic 3DGS remains unexplored.
- Pruning ratios (80%/30%) are empirically set and might require adjustment for different scenes.
- For scenes with a large number of semi-transparent Gaussians, the acceleration effect of SnugBox might be limited.

## Related Work & Insights

- SnugBox's elliptical bounding box computation can be generalized to other scenarios requiring precise 2D Gaussian range estimation (such as collision detection).
- The idea of Hessian reparameterization can be extended to other scenarios requiring sensitivity computation during training (such as NAS and structured pruning).
- Although similar in concept to StopThePop for exact tile intersection, Speedy-Splat is faster and incurs lower computational overhead.

## Rating

⭐⭐⭐⭐ — Outstanding engineering depth with solid theoretical analysis and CUDA implementation. The 6.71× speedup is highly practical without severely sacrificing quality. The plug-and-play, zero-accuracy-loss speedup of SnugBox and the Hessian reparameterization are both highly valuable independent contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[CVPR 2025\] SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)
- [\[CVPR 2025\] Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration](dr_splat_directly_referring_3d_gaussian_splatting_via_direct_language_embedding_.md)

</div>

<!-- RELATED:END -->
