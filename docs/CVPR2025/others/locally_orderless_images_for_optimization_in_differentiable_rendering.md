---
title: >-
  [Paper Note] Locally Orderless Images for Optimization in Differentiable Rendering
description: >-
  [CVPR 2025][differentiable rendering] This work proposes an inverse rendering optimization method that leverages local histogram matching within a 3D scale space (inner scale $\sigma$, tonal scale $\beta$, and extent scale $\alpha$) of Locally Orderless Images (LOIs). It expands the support range of sparse gradients without modifying differentiable renderers, effectively avoiding local optima.
tags:
  - "CVPR 2025"
  - "differentiable rendering"
  - "inverse rendering"
  - "locally orderless images"
  - "histogram matching"
  - "Wasserstein distance"
  - "scale space"
date: 2026-05-08
content_hash: 7914406f20f12f1e
---

# Locally Orderless Images for Optimization in Differentiable Rendering

**Conference**: CVPR 2025  
**arXiv**: [2503.21931](https://arxiv.org/abs/2503.21931)  
**Code**: To be confirmed  
**Area**: Others  
**Keywords**: differentiable rendering, inverse rendering, locally orderless images, histogram matching, Wasserstein distance, scale space

## TL;DR

This work proposes an inverse rendering optimization method that leverages local histogram matching within a 3D scale space (inner scale $\sigma$, tonal scale $\beta$, and extent scale $\alpha$) of Locally Orderless Images (LOIs). It expands the support range of sparse gradients without modifying differentiable renderers, effectively avoiding local optima.

## Background & Motivation

**Background**: Differentiable rendering iteratively optimizes scene parameters through analysis-by-synthesis. However, for parameters causing motion in image space (e.g., light source positions and geometric positions), the image gradients are extremely sparse (non-zero only at silhouette boundaries).

**Limitations of Prior Work**: Proxy gradient methods (e.g., topological, Lagrangian, and variational derivatives) are computationally expensive, support only implicit geometry, or handle only primary light transport effects. Multi-resolution pyramid matching has proven unreliable.

**Key Challenge**: Standard RGB gradients vanish (reduce to zero) when far from the target, whereas proxy gradient schemes introduce additional constraints and computational overhead.

**Key Insight**: Instead of modifying the renderer and gradient computation, this work alters the matching objective—shifting from pixel-value matching to local histogram matching.

**Core Idea**: Treat the image as a "family of local histograms" rather than a pixel-value function. By matching distributions instead of means in three orthogonal scale spaces, this preserves the gradient-expanding effect of spatial smoothing while maintaining distributional modalities.

## Method

### Overall Architecture

1. Rendered image $\mathcal{I}(x;\theta)$
2. Construct the Locally Orderless Image representation $\mathcal{H}(x, k; \theta, \sigma, \beta, \alpha)$ in three scale spaces
3. Compute the Wasserstein distance between the local histograms of the rendered image and the reference image
4. Sum over all scales to obtain the total error $\mathcal{E}_{\text{total}}(\theta) = \sum_{\alpha,\beta,\sigma} \mathcal{E}(\theta, \alpha, \beta, \sigma)$
5. Backpropagate to update the scene parameters $\theta$

### Key Designs

**1. Inner Scale Space (Inner Scale, $\sigma$) — Resolution Blurring**
- **Function**: Apply multi-resolution blurring to the rendered image using a Gaussian filter $G(x;\sigma)$: $\mathcal{I}(x;\theta,\sigma) = (G * I)(x;\sigma)$.
- **Mechanism**: Larger scales turn image features into larger overlapping regions, generating non-zero gradients. This behaves similarly to Gaussian pyramids but operates in a continuous scale space.
- **Design Motivation**: When the target and initialization do not overlap, gradients are zero at standard resolution; blurring allows them to overlap and produce gradient signals. However, pure blurring alters local appearance and suppresses high-frequency details.

**2. Tonal Scale Space (Tonal Scale, $\beta$) — Intensity Uncertainty Modeling**
- **Function**: Relax the radiance value of each pixel from a deterministic value into a probability distribution $\mathcal{P}(x, k; \theta, \sigma, \beta) = \frac{1}{\sqrt{2\pi\beta^2}} \exp(-\frac{(k-\mathcal{I}(x;\theta,\sigma))^2}{2\beta^2})$.
- **Mechanism**: For a given intensity $k$, $\mathcal{P}(x,k)$ acts as a "soft isophote," separating pixels of different intensities into independent image layers, preventing intensity confusion in multi-object scenes.
- **Design Motivation**: Monte Carlo rendering and sensor noise introduce inherent uncertainties to radiance values. Tonal separation ensures that objects with different appearances do not interfere with each other during optimization.

**3. Extent Scale Space (Extent Scale, $\alpha$) — Histogram Spatial Integration**
- **Function**: Integrate the local probability distribution using a spatial window $A(x;\alpha)$: $\mathcal{H}(x, k; \theta, \sigma, \beta, \alpha) = \int A(x-y;\alpha) \mathcal{P}(y, k) dy$.
- **Mechanism**: Unlike the inner scale which directly averages pixel values, the extent scale averages histogram contributions—thereby preserving the modal information of the distribution. Even at coarse scales, the peaks of the intensity distribution remain stable.
- **Design Motivation**: Gaussian pyramids lose the multi-modal characteristics of local appearances at coarse scales (retaining only the mean), whereas histogram integration preserves the full distribution and is more robust to noise.

### Loss & Training

- Uses the 1D Wasserstein distance (sum of pointwise CDF errors): $\mathcal{E} = \int \int [\text{cdf}_{\mathcal{H}'}(x,k) - \text{cdf}_{\mathcal{H}^{gt}}(x,k)]^{1/p} dk dx$
- The total error is the sum over all scales: $\mathcal{E}_{\text{total}} = \sum_{\alpha,\beta,\sigma} \mathcal{E}(\theta,\alpha,\beta,\sigma)$
- Typical parameters: $\alpha=[1,5,15]$, $\sigma=[1,5,15,45]$, $\beta=0.125$, $p=1$
- Performs optimization via standard gradient descent, compatible with any differentiable renderer.

## Key Experimental Results

### Main Results — 2D Differentiable Vectorization (Recovering $n$ Disc Positions)

| Method | $n=4$ PSNR | $n=16$ PSNR | $n=64$ PSNR | $n=256$ PSNR |
|---|---|---|---|---|
| Gaussian Pyramid | 25.05 | 19.83 | 15.34 | 9.60 |
| MS-SSIM | 27.44 | 21.41 | 15.42 | 9.83 |
| LPIPS | 27.79 | 20.94 | 17.04 | 15.2 |
| **Ours** | **30.40** | **33.55** | **28.23** | **21.57** |

### Comparison with Parameter-Space Blurring Methods

| Scene | PSNR (PRDPT) | PSNR (Ours) |
|---|---|---|
| Reflective Sphere Light Source Recovery | Local Optima | **Global Optima** |
| Multi-Object Position Recovery | Partially Successful | **Higher Success Rate** |

### Key Findings

1. **Ours leads significantly in all scales of disc recovery tasks**: at $n=256$, PSNR is 21.57 (Ours) vs 15.2 (LPIPS) vs 9.60 (GP), with the margin widening as the task difficulty increases.
2. **Gaussian Pyramids are unreliable in complex scenes**: mean matching fails to distinguish between objects with different appearances, frequently trapping the optimization in local optima in multi-target scenes.
3. **The LOI method is complementary to parameter-space blurring (PRDPT)**: they can be combined in complex light transport scenes.
4. **Robust to noise**: histogram distributions maintain modal stability under noise, which is an inherent property of kernel density estimation.
5. **Solves inverse problems requiring "long-range" feature matching using standard RGB gradients alone**.

## Highlights & Insights

- Elegant theoretical framework: introduces Koenderink & van Doorn's Local Orderless Image (LOI) theory to the inverse rendering domain.
- The design of three orthogonal scale spaces has clear physical meanings and complementarities.
- The method is completely decoupled from renderers—it does not alter renderer gradient computation, only the matching objective.
- The 1D Wasserstein distance has a closed-form solution, making the implementation simple and efficient.
- Can be combined with other optimization methods (such as variational optimization).

## Limitations & Future Work

- Multi-scale parameter selection (discrete values of $\alpha, \beta, \sigma$) requires manual configuration; adaptive strategies could be explored.
- The number and width of histogram bins affect performance, requiring tuning.
- Large-scale inverse rendering problems, such as 3D scene reconstruction, remain unexplored.
- The capability to handle topological changes (e.g., objects appearing/disappearing) was not analyzed.
- Computational overhead: multi-scale histogram construction and Wasserstein distance calculations increase the cost per iteration.

## Related Work & Insights

- Complementary to PRDPT (parameter-space blurring): LOI operates in the image space while PRDPT operates in the parameter space, allowing them to be combined.
- Koenderink & van Doorn's Locally Orderless Image theory was originally used for image topology analysis; this work elegantly reinterprets it as an optimization tool for inverse rendering.
- Insight: The concept of applying scale-space theory from signal processing to deep learning optimization (such as loss function design) is highly worth generalising.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Innovatively introduces classical image theory to differentiable rendering, with an elegantly designed three-scale framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers vectorization, ray tracing, and rasterization renderers, including experiments on real-world data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Progressively builds intuition starting from 1D examples, closely integrating theory and experiments.
- **Value**: ⭐⭐⭐⭐ Provides a new paradigm for inverse rendering optimization that is complementary to existing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiffBMP: Differentiable Rendering with Bitmap Primitives](../../CVPR2026/others/diffbmp_differentiable_rendering_with_bitmap_primitives.md)
- [\[CVPR 2025\] TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering](tensoflow_tensorial_flow-based_sampler_for_inverse_rendering.md)
- [\[CVPR 2025\] NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics](neisf_neural_incident_stokes_field_for_polarized_inverse_rendering_of_conductors.md)
- [\[NeurIPS 2025\] Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch](../../NeurIPS2025/others/scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](instance-wise_supervision-level_optimization_in_active_learning.md)

</div>

<!-- RELATED:END -->
