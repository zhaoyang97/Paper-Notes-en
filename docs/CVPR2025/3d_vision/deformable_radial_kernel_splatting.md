---
title: >-
  [Paper Note] Deformable Radial Kernel Splatting
description: >-
  [CVPR 2025][3D Vision][Gaussian Splatting] This paper proposes Deformable Radial Kernels (DRK) to generalize traditional Gaussian splatting. By leveraging learnable radial basis functions, $L_1$/$L_2$ norm blending, and edge-sharpening mechanisms, it achieves higher quality 3D scene rendering with fewer primitives.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Gaussian Splatting"
  - "Deformable Kernels"
  - "Novel View Synthesis"
  - "3D Scene Representation"
  - "Rasterization"
date: 2026-05-08
content_hash: 41537aba6362205b
---

# Deformable Radial Kernel Splatting

**Conference**: CVPR 2025  
**arXiv**: [2412.11752](https://arxiv.org/abs/2412.11752)  
**Code**: [https://yihua7.github.io/DRK-web/](https://yihua7.github.io/DRK-web/)  
**Area**: 3D Vision  
**Keywords**: Gaussian Splatting, Deformable Kernels, Novel View Synthesis, 3D Scene Representation, Rasterization

## TL;DR

This paper proposes Deformable Radial Kernels (DRK) to generalize traditional Gaussian splatting. By leveraging learnable radial basis functions, $L_1$/$L_2$ norm blending, and edge-sharpening mechanisms, it achieves higher quality 3D scene rendering with fewer primitives.

## Background & Motivation

Although 3D Gaussian Splatting (3DGS) has achieved immense success, Gaussian kernels suffer from three intrinsic limitations:
1. **Radial Symmetry Restriction**: Projecting onto screen space always yields ellipses, making it inefficient to represent diverse shapes such as rectangles and triangles.
2. **Smooth Boundary Constraint**: The $L_2$ norm only produces conic-section boundaries, which makes representing straight edges difficult.
3. **Scale-Sharpness Coupling**: In Gaussian distributions, the decay rate and spatial extent are coupled through the covariance matrix. Sharp features require narrow distributions, making it hard to simultaneously capture sharp transitions and large spatial extents.

Consequently, even simple basic shapes like triangles and rectangles require thousands of Gaussian primitives to approximate, leading to over-parameterization.

## Method

### Overall Architecture

DRK is a novel 2D plane-based primitive that extends traditional 2DGS. Each DRK is defined by a parameter set $\Theta=\{\mu, q, s_k, \theta_k, \eta, \tau, o, sh\}$, where $\{s_k, \theta_k\}_{k=1}^K$ control the shape, $\eta$ controls the boundary curvature, and $\tau$ controls the sharpness. The rendering pipeline is based on 3DGS, augmented with polygon clipping and sorting-cache strategies.

### Key Designs

1. **Learnable Radial Basis Functions**:
    - **Function**: Define the kernel shape using $K$ control points, overcoming the radial symmetry limitation of Gaussians.
    - **Mechanism**: Each control point is represented in polar coordinates $(s_k, \theta_k)$, where $s_k$ is the radial length and $\theta_k$ is the polar angle. For any point $(u,v)$, linear interpolation with cosine weights is performed between adjacent radial bases: $\alpha = o \cdot \exp(-\frac{r_2^2}{2}(\frac{1+\cos(\Delta\theta_k)}{2s_k^2} + \frac{1-\cos(\Delta\theta_k)}{2s_{k+1}^2}))$
    - **Design Motivation**: When $K=4$, angles are $k\pi/2$, and the relative axial scales are identical, it degenerates to a standard 2D Gaussian, ensuring backward compatibility.

2. **$L_1$/$L_2$ Norm Blending**:
    - **Function**: Achieve continuous control from curved to straight boundaries.
    - **Mechanism**: Introduce a blending weight $\eta \in (0,1)$, making the complete kernel function $\alpha = o \cdot \exp(-\frac{1}{2}(\eta r_1^2 + (1-\eta)\frac{r_2^2}{\bar{s}^2}))$. The $L_1$ norm is calculated via the inverse transform of adjacent endpoints, mapping its diamond-shaped unit ball to straight line segments between endpoints.
    - **Design Motivation**: The $L_2$ norm can only produce conic curves, whereas the $L_1$ norm can generate straight edges. Blending both allows flexible representation of linear edges commonly found in man-made environments.

3. **Edge-Sharpening Function**:
    - **Function**: Decouple the spatial extent and edge sharpness of the kernel.
    - **Mechanism**: Introduce a piecewise linear mapping function $\Psi(g)$ controlled by a sharpening coefficient $\tau \in (-1,1)$ to remap density values toward 0 or 1, producing sharper edge transitions while maintaining the spatial extent. The final opacity is formulated as $\alpha = o \cdot \Psi(g)$.
    - **Design Motivation**: In Gaussian kernels, sharp features require narrow distributions, preventing the simultaneous representation of large coverage and sharp edges; $\Psi$ decouples these two aspects.

### Rendering Optimization

- **Low-pass Filtering**: A view-dependent low-pass filter $\tilde{\alpha} = \max(\alpha, o \cdot \exp(\cdot))$ is adopted to scale the filter size according to the view cosine, preventing extremely small primitives from overfitting to a single training view.
- **Polygon Clipping**: Project the radial basis endpoints to form a polygon, allowing precise determination of whether a tile intersects with the kernel. This is more efficient than traditional AABB methods.
- **Sorting Cache**: Replace center-depth sorting with ray-plane intersection distance $r_t$ and maintain an 8-element sorted array, resolving sorting inconsistencies and popping artifacts when multiple kernels overlap.

## Key Experimental Results

### Main Results (DiverseScene Dataset)

| Method | PSNR↑ | LPIPS↓ | SSIM↑ | Primitives↓ |
|------|------|----------|------|------|
| 2D-GS | 33.92 | 0.0881 | 0.9514 | 359K |
| 3D-GS | 34.41 | 0.0861 | 0.9621 | 336K |
| 3D-HGS | 35.68 | 0.0637 | 0.9521 | 373K |
| GES | 35.05 | 0.0804 | 0.9634 | 330K |
| DRK | **37.58** | **0.0564** | **0.9752** | 260K |
| DRK (S2) | 35.03 | 0.0823 | 0.9637 | **42K** |

### Ablation Study

| Configuration | PSNR | LPIPS | Description |
|------|---------|------|------|
| DRK (S2) | 35.03 | 0.0823 | Extremely sparse, only 42K primitives |
| DRK (S1) | 36.62 | 0.0668 | Medium density, 109K primitives |
| DRK (Full) | 37.58 | 0.0564 | Full model, 260K primitives |

### Key Findings

- DRK substantially outperforms 3DGS, 2DGS, 3D-HGS, and GES across all rendering quality metrics.
- The extremely sparse version, DRK (S2), achieves comparable quality to GES using only **42K** primitives (1/8 of 3DGS).
- The model size can be reduced from 79.7MB (3DGS) to 12.3MB (DRK-S2).
- On the Mip-NeRF360 unbounded scenes, DRK shows a significant advantage in perceptual quality (LPIPS, SSIM).
- A single DRK primitive can flexibly model various shapes, such as rectangles, triangles, and ellipses, which would otherwise require hundreds of Gaussians.

## Highlights & Insights

- **Generalization from Specific to General**: Demonstrates that 2D Gaussian are a special case of DRK ($K=4$, symmetric angles), rendering the new method naturally backward-compatible with existing works.
- **Elegant Design of $L_1$/$L_2$ Blending**: Employs a single scalar $\eta$ to continuously control the transition from curved to straight boundaries while maintaining differentiability.
- **DiverseScene Dataset Contribution**: A newly created evaluation set covering texture, geometry, specularities, and large-scale scenes, filling the blank in scene diversity evaluation.
- **Pareto Frontier of Efficiency and Quality**: Different sparsity variants of DRK form a Pareto frontier that consistently outperforms existing methods.

## Limitations & Future Work

- The rendering speed of the full DRK (77.5 FPS) is lower than that of 3DGS (247 FPS) due to the more complex kernel calculations.
- Overfitting may occur in far regions of unbounded scenes due to insufficient supervisory signals.
- The choice of the number of radial bases $K$ is a hyperparameter; adaptive adjustment could be considered.
- The potential of DRK in downstream applications such as dynamic scenes and generative tasks remains unexplored.

## Related Work & Insights

- GES controls sharpness by adjusting exponential values but retains rotational symmetry; DisC-GS and 3D-HGS use cutting techniques to handle discontinuities but are still constrained by Gaussian smoothness. DRK fundamentally addresses the shape limitations.
- Insight: In 3D representation, the design space of kernel functions is far larger than that of Gaussians, and learnable shape parameterization is a promising direction.
- It is connected to concurrent work 3D-CS (which uses smooth convex shapes), but DRK is more flexible.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Fundamentally generalizes the Gaussian kernel with elegant mathematical derivation and clever backward-compatibility proofs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on both the self-built dataset and Mip-NeRF360, with adequate analysis of variants with different sparsities, though the ablation study could be deeper.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Equations are clearly derived and the illustrations are excellent, especially the highly intuitive shape comparison visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Opens up a new direction beyond Gaussian kernels, having a profound impact on 3D representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PRaDA: Projective Radial Distortion Averaging](prada_projective_radial_distortion_averaging.md)
- [\[CVPR 2025\] RDD: Robust Feature Detector and Descriptor Using Deformable Transformer](rdd_robust_feature_detector_and_descriptor_using_deformable_transformer.md)
- [\[ECCV 2024\] Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting](../../ECCV2024/3d_vision/per-gaussian_embedding-based_deformation_for_deformable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting](feat2gs_probing_visual_foundation_models_with_gaussian_splatting.md)
- [\[CVPR 2025\] Geometry Field Splatting with Gaussian Surfels](geometry_field_splatting_with_gaussian_surfels.md)

</div>

<!-- RELATED:END -->
