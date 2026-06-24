---
title: >-
  [Paper Note] Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions
description: >-
  [ICCV 2025][Image Restoration][Adaptive Convolution] This paper proposes a metric-geometric perspective that unifies existing adaptive convolution variants (standard, dilated, shifted, and deformable), and introduces Metric Convolution based on unit-ball sampling of an explicit Randers metric, achieving superior geometric regularization and generalization with substantially fewer parameters.
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Adaptive Convolution"
  - "Metric Geometry"
  - "Finsler Metric"
  - "Deformable Convolution"
  - "Denoising"
date: 2026-05-08
content_hash: 5763cd41a8f6fa38
---

# Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions

**Conference**: ICCV 2025
**arXiv**: [2406.05400](https://arxiv.org/abs/2406.05400)  
**Code**: [GitHub](https://github.com/Tommoo/MetricConvolutions)  
**Area**: Image Restoration
**Keywords**: Adaptive Convolution, Metric Geometry, Finsler Metric, Deformable Convolution, Denoising

## TL;DR

This paper proposes a metric-geometric perspective that unifies existing adaptive convolution variants (standard, dilated, shifted, and deformable), and introduces Metric Convolution based on unit-ball sampling of an explicit Randers metric, achieving superior geometric regularization and generalization with substantially fewer parameters.

## Background & Motivation

Standard convolutions are a cornerstone of deep learning, yet their fixed isotropic $k \times k$ kernel shape limits adaptability to deformed objects and complex spatial transformations. The community has proposed numerous variants:

- **Dilated Convolutions**: uniformly scale sampling intervals to enlarge the receptive field, but lack data adaptivity.
- **Spatial Transformer Networks (STN)**: learn globally parameterized transformations, but are constrained to predefined transformation families.
- **Active Convolution**: learns anisotropic offsets shared across all spatial locations.
- **Deformable Convolution (DCN)**: learns per-pixel, per-kernel-position offsets, offering flexibility but lacking theoretical constraints.

**Key Challenge**: Although empirically effective, these methods lack a unified theoretical framework for understanding their capabilities and limitations. The various deformation strategies appear as a collection of performance-driven tricks without intrinsic connections.

The key insight of this paper is to **treat an image as a two-dimensional manifold equipped with a metric**. Under this view, the kernel sampling neighborhood of each convolution variant can be reinterpreted as unit-ball sampling of some implicit metric. This observation motivates two directions: (1) providing geometric interpretability for existing convolutions, and (2) designing a new Metric Convolution based on an explicit metric.

## Method

### Overall Architecture

1. **Unifying Theory**: Proves that all existing convolutions (standard, dilated, shifted, deformable) can be expressed as weighted signal averages over the unit ball of some implicit metric.
2. **Metric Convolution**: Explicitly constructs a signal-dependent parameterized metric (Randers metric) and samples kernel positions from its unit ball.
3. Can be used as a single-layer denoising filter or embedded into a CNN to replace standard convolutional layers.

### Key Designs

1. **Unified Metric Theory (Theorems 1–2)**:

    - Any standard/dilated/shifted/deformable convolution can be expressed as $(f*g)(x) = \int_{\Delta_x} f(x+y) g(y) dm_x(y)$, where $\Delta_x$ is the $x$-dependent local support.
    - A metric is uniquely determined by its unit tangent ball (Theorem 2).
    - Consequently, all convolution variants are essentially weighted averages over the unit ball of an implicit metric.

2. **Randers Metric Parameterization**:

    - Riemannian metric: $R_x(u) = \sqrt{u^\top M(x) u}$, defined by a 2×2 positive definite matrix $M$.
    - Randers metric (a subclass of Finsler metrics): $F_x(u) = \sqrt{u^\top M(x) u} + \omega(x)^\top u$, augmented with a linear drift term $\omega$ to allow **asymmetric neighborhoods**.
    - Asymmetry is particularly useful for edge preservation: near object boundaries, the neighborhood should not extend across into the background.

3. **Unit Tangent Ball (UTB) Sampling**:

    - By positive homogeneity of the metric, the point on the unit ball at angle $\theta$ is $y_x(\theta, \gamma) = \frac{1}{F_x^\gamma(u_\theta)} u_\theta$.
    - The UTB is discretized in polar coordinates to yield $k^2$ sample points.
    - Key advantage: metric parameters $\gamma = (M, \omega)$ require only 5–7 values (3 for Cholesky decomposition + 2 for $\omega$), whereas deformable convolution requires $2k^2$ offset parameters.

4. **Obtaining Metric Parameters**:

    - Heuristic design: eigenvectors of $M$ are set to the image gradient $\nabla f$ and its orthogonal direction, with eigenvalues controlling anisotropy.
    - Learnable approach: metric parameters are predicted from the input signal via intermediate standard convolutions, preserving shift equivariance.

### Loss & Training

- MSE loss is used for denoising tasks.
- For classification, standard 3×3 convolutions in layers 2–4 of ResNet18 are replaced with Metric Convolutions.
- Both fixed kernel weights (FKW) and learnable kernel weights (LKW) modes are supported.
- Adam optimizer is used with task-specific learning rate schedules.

## Key Experimental Results

### Main Results

Denoising comparison on BSDS300 and PascalVOC (learned filter, $k=5$, noise $\sigma_n=0.1$):

| Method | MSE (BSDS300) | MSE (PascalVOC) | Parameter Efficiency |
|--------|--------------|-----------------|----------------------|
| Deformable (FKW) | 1.12e-4 | 8.59e-5 | $2k^2$ channels |
| Deformable (LKW) | 1.72e-4 | 1.02e-4 | $2k^2$ channels |
| **Metric UTB ε=0.1 (FKW)** | **1.19e-4** | **1.01e-4** | **5 channels** |
| **Metric UTB ε=0.1 (LKW)** | **1.64e-4** | **1.06e-4** | **5 channels** |

CNN Classification (ResNet18, CIFAR-10, LKW-TL):

| Method | Top-1 Accuracy | Std. Dev. |
|--------|---------------|-----------|
| Standard Conv | 92.64% | ±0.18% |
| Deformable Conv | 93.10% | ±0.17% |
| Shifted Conv | 92.58% | ±0.28% |
| **Metric UTB (Ours)** | **93.07%** | **±0.13%** |

### Ablation Study

Generalization gap $\delta_{\text{MSE}}$ for single-image denoising across kernel sizes (noise $\sigma_n=0.3$):

| Method | k=5 | k=11 | k=31 | k=51 | k=121 |
|--------|-----|------|------|------|-------|
| Deformable | 265 | 74 | 28 | 18 | 6.6 |
| **Metric UTB (ε=0.9)** | **1.1** | **0.9** | **1.1** | **0.8** | **1.2** |
| **Metric UTB (ε=0.1)** | **1.3** | **1.1** | **1.3** | **1.4** | **1.5** |

The generalization gap of deformable convolution grows rapidly with $k$ (overfitting), whereas Metric Convolution maintains a consistently low gap across all kernel sizes.

### Key Findings

- **Geometric priors provide strong regularization**: Metric CNN is nearly unaffected when trained from scratch (SC), whereas DCN and Shifted Conv suffer significant performance degradation.
- **Fixed kernel weights remain effective**: Metric CNN retains reasonable performance under FKW settings, whereas DCN degrades to near-random predictions.
- GradCAM visualizations show that Metric CNN focuses more accurately on relevant objects and semantically meaningful regions, rather than background.
- Asymmetric metrics (small $\varepsilon_\omega$) generally outperform symmetric ones, as they allow neighborhoods to extend asymmetrically along edges.

## Highlights & Insights

- The metric-geometric perspective elegantly unifies seemingly disparate convolution variants.
- Metric Convolution describes kernel position deformation with only 5–7 parameters, compared to $2k^2$ for DCN, offering exceptional parameter efficiency.
- The asymmetry introduced by Finsler/Randers metrics provides a natural advantage for edge preservation.
- Shift equivariance is rigorously proven (Theorem 3), establishing a solid theoretical foundation.

## Limitations & Future Work

- The Randers metric constrains the unit tangent ball to an ellipsoidal shape, precluding more complex convex geometries.
- The geodesic ball (UGB) variant is computationally prohibitive for practical CNN deployment.
- As with all non-standard convolutions, per-pixel offset storage is required, incurring high computational cost at large resolutions.
- Validation is currently limited to lower-resolution benchmarks; high-resolution data and more complex tasks remain to be explored.

## Related Work & Insights

- The modulation mechanism of Deformable Convolution v2 (DCNv2) can be interpreted as a non-uniform sampling probability distribution over the unit ball.
- InternImage scales large deformable convolutions toward foundation models; the parameter efficiency of Metric Convolution may prove advantageous in this direction.
- Anisotropic convolutions on graphs and surfaces (e.g., the work of Boscaini et al.) are conceptually aligned with Metric Convolution.
- The asymmetry of the Randers metric also finds applications in classical vision tasks such as active contours.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The metric-geometric unifying perspective is highly original.
- Technical Depth: ⭐⭐⭐⭐⭐ — The theoretical framework is complete and rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-task validation with comprehensive ablations.
- Practical Value: ⭐⭐⭐ — Utility at high resolutions remains to be verified.
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICML 2026\] Degradation-Aware Metric Prompting for Hyperspectral Image Restoration](../../ICML2026/image_restoration/degradation-aware_metric_prompting_for_hyperspectral_image_restoration.md)
- [\[NeurIPS 2025\] Adaptive Discretization for Consistency Models](../../NeurIPS2025/image_restoration/adaptive_discretization_for_consistency_models.md)
- [\[ECCV 2024\] Efficient Cascaded Multiscale Adaptive Network for Image Restoration](../../ECCV2024/image_restoration/efficient_cascaded_multiscale_adaptive_network_for_image_restoration.md)

</div>

<!-- RELATED:END -->
