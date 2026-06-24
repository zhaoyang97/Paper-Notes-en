---
title: >-
  [Paper Note] TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering
description: >-
  [CVPR 2025][Inverse Rendering] This paper proposes TensoFlow, which learns a spatially and directionally aware importance sampler using Tensorial Normalizing Flow to replace fixed, predefined samplers (e.g., cosine-weighted, GGX) in inverse rendering. This significantly reduces the variance of Monte Carlo estimation of the rendering equation and improves the quality of material and illumination decomposition.
tags:
  - "CVPR 2025"
  - "Inverse Rendering"
  - "Importance Sampling"
  - "Normalizing Flow"
  - "Tensor Decomposition"
  - "Material Estimation"
date: 2026-05-08
content_hash: eeefb842b1da8586
---

# TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering

**Conference**: CVPR 2025  
**arXiv**: [2503.18328](https://arxiv.org/abs/2503.18328)  
**Code**: [https://github.com/fudan-zvg/tensoflow](https://github.com/fudan-zvg/tensoflow)  
**Area**: LLM Evaluation  
**Keywords**: Inverse Rendering, Importance Sampling, Normalizing Flow, Tensor Decomposition, Material Estimation

## TL;DR
This paper proposes TensoFlow, which learns a spatially and directionally aware importance sampler using Tensorial Normalizing Flow to replace fixed, predefined samplers (e.g., cosine-weighted, GGX) in inverse rendering. This significantly reduces the variance of Monte Carlo estimation of the rendering equation and improves the quality of material and illumination decomposition.

## Background & Motivation
1. **Background**: Inverse rendering aims to recover the geometry, material properties, and illumination of a scene from multi-view images. Physics-based rendering equations require Monte Carlo sampling to solve hemispherical integrals, and importance sampling is a key technique to reduce variance.
2. **Limitations of Prior Work**: Methods like NeRO and TensoSDF employ predefined fixed samplers (cosine-weighted for diffuse and GGX distribution for specular reflections). However, the distribution of the integrand in a scene varies highly along with spatial positions and directions. Fixed samplers fail to match this variation, leading to high variance and sub-optimal performance.
3. **Key Challenge**: An ideal importance sampler should match the shape of the integrand. However, the integrand is jointly determined by the BRDF, incident illumination, and geometric normals, and it varies across space and direction depending on the location—this necessitates a learnable, position-aware sampling distribution.
4. **Goal**: To learn a trainable importance sampler capable of simultaneously sensing spatial positions and reflection directions.
5. **Key Insight**: Normalizing flows naturally support PDF inference and sampling, and can model arbitrarily complex distributions.
6. **Core Idea**: To formulate a normalizing flow using piecewise-quadratic coupling layers conditioned on spatial features from tensor decomposition and reflection directions, thereby achieving a spatially and directionally adaptive importance sampler.

## Method

### Overall Architecture
TensoFlow operates in two stages: (1) Geometry Reconstruction: following TensoSDF's tensorial SDF to reconstruct scene geometry; (2) Material/Illumination Estimation (Core): parameterizing material properties (albedo, metallic, roughness) using a tensorial encoder, while learning an importance sampler via a tensorial normalizing flow. During rendering equation evaluation, incident directions are sampled from the learned sampler, and the PDF is inferred.

### Key Designs

1. **Flow-based Sampler**

    - **Function**: Replaces fixed cosine/GGX samplers to provide a learnable importance sampling distribution.
    - **Mechanism**: Represents the incident direction $\omega$ as a uniformly distributed variable $z \sim \mathcal{U}(0,1)^2$ transformed by the normalizing flow, i.e., $\omega = h(z)$. The normalizing flow consists of multiple piecewise-quadratic coupling layers $h = h_n \circ \cdots \circ h_1$, where each coupling layer keeps one dimension constant and transforms the other dimension via a piecewise-quadratic CDF. The triangular Jacobian matrix enables efficient determinant computation. It supports bidirectional operations: sampling (forward of $h$) and PDF inference ($h^{-1}$), meeting the dual requirements of importance sampling in the rendering equation. Modeling the half-vector $\omega_h$ instead of directly modeling $\omega_i$ yields better results.
    - **Design Motivation**: The shape of the integrand's distribution varies drastically across different scene locations (e.g., concentrated in specular regions, uniform in diffuse regions), which fixed distributions cannot adapt to.

2. **Tensorial Coupling Transform**

    - **Function**: Injects spatial and directional priors into each coupling layer of the normalizing flow.
    - **Mechanism**: Encodes spatial scene features using a Vector-Matrix tensor decomposition $V_f(x) = v_{f,k}^X \circ M_{f,k}^{YZ} \oplus v_{f,k}^Y \circ M_{f,k}^{XZ} \oplus v_{f,k}^Z \circ M_{f,k}^{XY}$, which is then concatenated with the reflection direction $\omega_r = 2(\omega_o \cdot n)n - \omega_o$ as the conditioning input to the coupling layer's internal network $m_i$. The network $m_i$ outputs the vertex values $\hat{V}$ and bin widths $\hat{W}$ of a piecewise-linear PDF, which are processed via softmax/normalization to guarantee a valid probability distribution. The $K+1$ vertices define the piecewise-linear PDF, which integrates to a piecewise-quadratic CDF.
    - **Design Motivation**: The integrand of the rendering equation is determined by $f(\omega_o, \omega_i, x) \cdot L_i(\omega_i, x) \cdot (\omega_i \cdot n)$, whose shape depends on both the surface position $x$ and the reflection direction $\omega_r$. Consequently, the sampler must be spatially and directionally aware.

3. **Cross-Entropy Training Optimization**

    - **Function**: Guides the sampler distribution to approximate the normalized shape of the integrand.
    - **Mechanism**: Minimizes the cross-entropy between the integrand $I(\omega_i, \omega_o, x)$ and the sampler PDF $q(\omega_i)$: $\mathcal{L}_{ce} = \mathbb{E}[-\frac{I(\omega_i, \omega_o, x)}{\hat{q}(\omega_i)} \log q(\omega_i)]$. A "frozen copy" strategy is employed, where the normalizing flow used for sampling is a periodic snapshot (updated every $N_{update}$ iterations) of the training version to avoid training instability. Separate samplers are learned for diffuse and specular reflections.
    - **Design Motivation**: The optimal solution of cross-entropy happens to be the distribution where $q(\omega_i) \propto I(\omega_i, \omega_o, x)$, representing the theoretically optimal importance sampling.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_c + \lambda_{ce}^d \mathcal{L}_{ce}^d + \lambda_{ce}^s \mathcal{L}_{ce}^s + \mathcal{L}_{reg}$
- RGB rendering loss $\mathcal{L}_c$ supervises material parameters
- Cross-entropy loss $\mathcal{L}_{ce}^{d/s}$ optimizes diffuse/specular samplers
- Material regularization loss $\mathcal{L}_{reg}$

## Key Experimental Results

### Main Results (TensoSDF Synthetic Dataset)

| Method | Sampler Type | Albedo MAE↓ | Roughness MAE↓ | Relighting PSNR↑ |
|------|-----------|-------------|----------------|-----------------|
| TensoSDF | Fixed (cos+GGX) | ~0.045 | ~0.12 | ~28.5 |
| NeRO | Fixed (cos+GGX) | ~0.050 | ~0.14 | ~27.0 |
| **TensoFlow** | Learnable | **~0.035** | **~0.09** | **~30.0** |

### Ablation Study

| Configuration | Relighting PSNR↑ | Description |
|------|-----------------|------|
| Full TensoFlow | ~30.0 | Full model |
| w/o Spatial condition $V_f$ | ~28.8 | Sampler degenerates to direction-aware |
| w/o Direction condition $\omega_r$ | ~29.2 | Sampler degenerates to space-aware |
| Fixed cosine+GGX sampler | ~28.5 | Degenerates to TensoSDF |
| Directly modeling $\omega_i$ instead of $\omega_h$ | ~29.3 | Half-vector modeling is more effective |

### Key Findings
- Learnable samplers significantly reduce the variance of rendering equation estimation given the same number of samples, directly improving material decomposition accuracy.
- Spatial and directional conditions contribute independently, but their combination achieves the best performance.
- Piecewise-quadratic coupling layers are more expressive than other coupling transforms (e.g., affine coupling).
- It continuously outperforms baseline fixed samplers on real-world datasets.

## Highlights & Insights
- **"Learning the sampler instead of using a fixed one"** is the core contribution of this paradigm shift. In traditional computer graphics, samplers are handcrafted. This work introduces it as a learnable component for the first time, leveraging the theoretical advantages of normalizing flows (supporting both sampling and PDF inference).
- **Combining tensor decomposition for spatial encoding** with normalizing flows is highly intuitive: tensor decomposition provides efficient spatial queries, while normalizing flows offer flexible distribution modeling, complementing each other.
- **The frozen copy training strategy** cleverly decouples "which distribution to sample from" and "which distribution to optimize", avoiding the instability associated with bootstrapping.

## Limitations & Future Work
- The extra normalizing flow increases model complexity and training time.
- Currently, it only supports scenes represented by SDFs; integration with 3DGS remains an open problem.
- The number of bins $K$ in the piecewise-quadratic coupling layer is a hyperparameter.
- Integrating Multiple Importance Sampling (MIS) with learnable samplers is a promising direction for future exploration.

## Related Work & Insights
- **vs NeRO/TensoSDF**: These are also inverse rendering methods but employ fixed cosine+GGX samplers, which suffer from high variance under complex illumination. In contrast, the proposed learnable sampler adaptively matches the integrand.
- **vs NeILF/TensoIR**: These methods utilize stratified uniform sampling, which is less efficient and requires a larger number of sample points.
- **vs Neural Importance Sampling (Müller 2019)**: While that work learns samplers in forward rendering, this paper is the first to introduce a similar concept to inverse rendering.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce a learnable importance sampler in inverse rendering, with a solid theoretical foundation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid evaluation on synthetic and real datasets with detailed ablation studies, though quantitative comparisons on real-world scenes could be more extensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation and very clear description of the methodology.
- Value: ⭐⭐⭐⭐ Opens up a new direction for sampling strategies in inverse rendering, though the scope of application is limited by SDF representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics](neisf_neural_incident_stokes_field_for_polarized_inverse_rendering_of_conductors.md)
- [\[CVPR 2025\] Locally Orderless Images for Optimization in Differentiable Rendering](locally_orderless_images_for_optimization_in_differentiable_rendering.md)
- [\[CVPR 2025\] UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation](uniphy_learning_a_unified_constitutive_model_for_inverse_physics_simulation.md)
- [\[NeurIPS 2025\] Adjoint Schrödinger Bridge Sampler](../../NeurIPS2025/others/adjoint_schrödinger_bridge_sampler.md)
- [\[CVPR 2026\] DiffBMP: Differentiable Rendering with Bitmap Primitives](../../CVPR2026/others/diffbmp_differentiable_rendering_with_bitmap_primitives.md)

</div>

<!-- RELATED:END -->
