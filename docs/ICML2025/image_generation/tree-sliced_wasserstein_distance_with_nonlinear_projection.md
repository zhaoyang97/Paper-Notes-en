---
title: >-
  [Paper Note] Tree-Sliced Wasserstein Distance with Nonlinear Projection
description: >-
  [ICML2025][Image Generation][Optimal Transport] This paper proposes the Tree-Sliced Wasserstein (TSW) distance under a nonlinear projection framework. By replacing linear projections with Circular and Spatial nonlinear Radon transforms, the proposed method preserves the well-definedness and injectivity of the metric while significantly outperforming existing SW and TSW variants on tasks such as gradient flows, self-supervised learning, and generative models.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Optimal Transport"
  - "Tree-Sliced Wasserstein Distance"
  - "Nonlinear Projection"
  - "Radon Transform"
  - "Probability Metric"
  - "Spherical Metric"
date: 2026-05-08
content_hash: a09320e47213a5b9
---

# Tree-Sliced Wasserstein Distance with Nonlinear Projection

**Conference**: ICML2025  
**arXiv**: [2505.00968](https://arxiv.org/abs/2505.00968)  
**Code**: [thanhqt2002/NonlinearTSW](https://github.com/thanhqt2002/NonlinearTSW)  
**Area**: Image Generation  
**Keywords**: Optimal Transport, Tree-Sliced Wasserstein Distance, Nonlinear Projection, Radon Transform, Probability Metric, Spherical Metric

## TL;DR
This paper proposes the Tree-Sliced Wasserstein (TSW) distance under a nonlinear projection framework. By replacing linear projections with Circular and Spatial nonlinear Radon transforms, the proposed method preserves the well-definedness and injectivity of the metric while significantly outperforming existing SW and TSW variants on tasks such as gradient flows, self-supervised learning, and generative models.

## Background & Motivation
**Optimal Transport (OT)** provides geometrically meaningful metrics on the space of probability measures, but suffers from high computational complexity. The **Sliced Wasserstein (SW) distance** utilizes the closed-form solution of one-dimensional OT to reduce complexity by projecting high-dimensional measures onto one-dimensional lines via the Radon transform.

The **Tree-Sliced method** is an alternative to SW:
- It replaces the one-dimensional line with a tree metric space.
- It introduces a splitting mechanism (splitting map) to distribute measures onto multiple connected lines.
- It better captures the topology of the integration domain while maintaining low computational overhead.

**Limitations of Prior Work**: The Tree-Sliced framework remains limited to linear projections (hyperplane integration domains), whereas various nonlinear projection enhancement schemes (Circular RT, Spatial RT, etc.) have already been introduced to improve SW.

**Core Problem**: Can the nonlinear projection framework be integrated into Tree-Sliced methods to achieve both the advantages of tree structures and the expressive power of nonlinear projections?

## Method

### Preliminaries

**Radon Transform**: $\mathcal{R}f(t, \theta) = \int_{\mathbb{R}^d} f(y) \cdot \delta(t - \langle y, \theta \rangle) dy$

**Generalized Radon Transform (GRT)**: Replaces the inner product $\langle y, \theta \rangle$ with a general function $g(y, \psi)$, projecting along hypersurfaces instead of hyperplanes.

**Spatial Radon Transform (SRT)**: First transforms the space using an injective continuous mapping $h: \mathbb{R}^d \to \mathbb{R}^{d_\theta}$, and then performs linear projection.

### 1. Generalization of Circular Radon Transform to Line Systems (CRTSL)

$$\mathcal{CR}^{\alpha}_{\mathcal{L},r} f(x_i + t \cdot \theta_i) = \int_{\mathbb{R}^d} f(y) \cdot \alpha(y, \mathcal{L})_i \cdot \delta(t - \|y - x_i - r\theta_i\|_2) \, dy$$

- Replaces the inner product projection with a Euclidean distance function, shifting the integration domain from hyperplanes to (hyper)spheres.
- $\alpha(y, \mathcal{L})_i$ is a splitting map that distributes the mass of function $f$ to each line of the line system.
- Parameter $r$ controls the center offset.

### 2. Generalization of Spatial Radon Transform to Line Systems (SRTSL)

$$\mathcal{H}^{\alpha}_{\mathcal{L}} f(x_i + t \cdot \theta_i) = \int_{\mathbb{R}^d} f(y) \cdot \alpha(h(y), \mathcal{L})_i \cdot \delta(t - \langle h(y) - x_i, \theta_i \rangle) \, dy$$

- First uses an injective mapping $h$ to transform data into a new space $\mathbb{R}^{d_\theta}$, and then performs Tree-Slicing in the new space.
- Choice of $h$: Component-wise odd polynomials $h(x_1,...,x_d) = (f_1(x_1),...,f_d(x_d))$ (e.g., $f_i(x)=x+x^3$), or a concatenated neural network $h(x) = (x, \phi(x))$.

### Guarantee of Injectivity

**Theorem 4.2**: If the splitting map $\alpha$ is $\text{E}(d)$-invariant, then CRTSL is injective.

**Theorem 4.3**: If the splitting map $\alpha$ is $\text{E}(d_\theta)$-invariant, then SRTSL is injective.

The chosen splitting map is a distance-based softmax: $\alpha(x, \mathcal{L})_l = \text{softmax}(\{d(x, \mathcal{L})_i\}_{i=1}^k)$, which naturally satisfies $\text{E}(d)$-invariance.

### 3. Definitions of New Distances

**CircularTSW**: $\text{CircularTSW}(\mu, \nu) = \int_{\mathbb{T}^d_k} \text{W}(\mathcal{CR}^{\alpha}_{\mathcal{L},r} f_\mu, \mathcal{CR}^{\alpha}_{\mathcal{L},r} f_\nu) \, d\sigma(\mathcal{L})$

**SpatialTSW**: $\text{SpatialTSW}(\mu, \nu) = \int_{\mathbb{T}^{d_\theta}_k} \text{W}(\mathcal{H}^{\alpha}_{\mathcal{L}} f_\mu, \mathcal{H}^{\alpha}_{\mathcal{L}} f_\nu) \, d\sigma(\mathcal{L})$

**Theorem 5.3**: Both CircularTSW and SpatialTSW are metrics on $\mathcal{P}(\mathbb{R}^d)$.

### 4. Computational Advantage of CircularTSW$_{r=0}$
When $r=0$, all support points have the same projection coordinates across the $k$ lines of the tree. Therefore, sorting needs to be performed only once (instead of $k$ times), reducing the complexity from $O(Lkn\log n)$ to $O(Ln\log n + Lkd_\theta n)$.

### 5. Spherical Extension (SpatialSTSW)
Generates the framework to measures on the sphere $\mathbb{S}^d$, using Spherical Trees and $\text{O}(d_\theta+1)$-invariant splitting maps.

## Key Experimental Results

### Computational Efficiency (Runtime Comparison)

| Method | Relative SW Speed |
|------|------------|
| SW (vanilla) | 1× |
| Db-TSW | Slower |
| CircularTSW | Slightly slower |
| **CircularTSW$_{r=0}$** | **Close to SW** |

CircularTSW$_{r=0}$ is the only Tree-Sliced method that can approach the speed of vanilla SW.

### Application Performance (Gradient Flows, Generative Models, etc.)
- In gradient flow experiments, both SpatialTSW and CircularTSW outperform SW and linear TSW in terms of convergence speed and quality.
- Significantly outperforms recent SW/TSW variants in denoising diffusion GANs and self-supervised learning tasks.
- On spherical data, SpatialSTSW outperforms Spherical SW and Spherical TSW.

### Key Findings
- Nonlinear projections significantly enhance the capturing capacity of Tree-Sliced distances, especially on high-dimensional data.
- CircularTSW$_{r=0}$ performs well under the tree system framework but poorly under the original sliced setting, demonstrating the necessity of the tree structure and splitting maps.
- Concatenating a neural network as $h(x)=(x,\phi(x))$ introduces learnable parameters, offering a flexible performance-computation trade-off.

## Highlights & Insights
1. **The combination of nonlinear projection and tree structure** is the core innovation of this work—both possess distinct advantages, and their combination yields significant improvements.
2. **Injectivity proofs for CRTSL and SRTSL** provide solid theoretical guarantees for the well-definedness of the metrics.
3. **The computational trick of CircularTSW$_{r=0}$** is elegant: leveraging the identical coordinates when $r=0$ avoids redundant sorting, achieving speed close to vanilla SW.
4. **Spherical extension** demonstrates the generality of the framework, which is not restricted to Euclidean spaces.
5. The splitting map employs distance-based softmax, which is simple, elegant, and automatically satisfies group invariance.

## Limitations & Future Work
1. The choice of the nonlinear projection function $h$ significantly impacts performance, yet a systematic guideline for selection is lacking.
2. Theoretically, the injectivity of CRTSL relies on the $\text{E}(d)$-invariance assumption; in practice, the softmax approximation might introduce errors.
3. Experiments are mainly validated on small-to-medium scale datasets; scalability to large-scale high-dimensional data (e.g., million-scale images) has not been sufficiently evaluated.
4. Theoretical and experimental comparisons with other OT approximation methods (such as Sinkhorn or entropic OT) are insufficient.

## Related Work & Insights
- **Tran et al., 2025c/b**: Pioneers of the Tree-Sliced framework, establishing the direct foundation of this work.
- **Kolouri et al., 2019**: Application of the Generalized Radon Transform in SW (Generalized SW).
- **Chen et al., 2022**: Spatial Radon Transform (Augmented SW).
- **Bonneel et al., 2015**: Classical Sliced Wasserstein distance.
- Insight: In metric design, the "projection method" and the "metric space structure" are two independently optimizable dimensions.

## Rating
- Novelty: ⭐⭐⭐⭐ (Combination of nonlinear projection and tree structure, with a complete theoretical framework)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive coverage of gradient flows, generative models, self-supervised learning, and spherical data)
- Writing Quality: ⭐⭐⭐⭐ (Mathematically rigorous and clear notation, though formulas are highly dense)
- Value: ⭐⭐⭐⭐ (Advances the frontier of Sliced OT; open-source code is reproducible)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Tree-Sliced Wasserstein Distance: A Geometric Perspective](tree-sliced_wasserstein_distance_a_geometric_perspective.md)
- [\[ICLR 2026\] Efficient Sliced Wasserstein Distance Computation via Adaptive Bayesian Optimization](../../ICLR2026/image_generation/efficient_sliced_wasserstein_distance_computation_via_adaptive_bayesian_optimiza.md)
- [\[ICML 2025\] Importance Sampling for Nonlinear Models](importance_sampling_for_nonlinear_models.md)
- [\[NeurIPS 2025\] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres](../../NeurIPS2025/image_generation/schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)
- [\[ICML 2025\] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning](local_manifold_approximation_and_projection_for_manifold-aware_diffusion_plannin.md)

</div>

<!-- RELATED:END -->
