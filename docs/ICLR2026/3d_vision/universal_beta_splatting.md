---
title: >-
  [Paper Note] Universal Beta Splatting
description: >-
  [3D Vision] This paper proposes Universal Beta Splatting (UBS), which generalizes 3D Gaussian Splatting to an N-dimensional anisotropic Beta kernel. By enabling per-dimension shape control, UBS unifies spatial geometry…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 20a245382e46ee39
---

# Universal Beta Splatting

- **Conference**: ICLR 2026
- **arXiv**: [2510.03312](https://arxiv.org/abs/2510.03312)
- **Code**: [Project Page](https://rongliu-leo.github.io/universal-beta-splatting/)
- **Area**: 3D Vision / Neural Rendering
- **Keywords**: 3D Gaussian Splatting, Beta Kernel, N-Dimensional, View-Dependent, Dynamic Scene, Real-Time Rendering

## TL;DR

This paper proposes Universal Beta Splatting (UBS), which generalizes 3D Gaussian Splatting to an N-dimensional anisotropic Beta kernel. By enabling per-dimension shape control, UBS unifies spatial geometry, view-dependent appearance, and scene dynamics within a single representation, achieving interpretable scene decomposition and state-of-the-art rendering quality.

## Background & Motivation

3D Gaussian Splatting (3DGS) achieves real-time rendering via explicit primitives, yet the fixed bell-shaped profile of the Gaussian kernel imposes fundamental limitations:

**Spatial dimensions**: Sharp boundaries require large numbers of small primitives, resulting in low efficiency.

**Angular dimensions**: View-dependent effects rely on additional spherical harmonic encoding (48 parameters), leading to a fragmented representation.

**Temporal dimensions**: Dynamic scenes require auxiliary deformation networks, increasing overall complexity.

**Core Insight**: Different scene properties demand different kernel behaviors—spatial geometry requires adaptive sharpness, angular appearance ranges from diffuse to specular, and temporal dynamics span from static to fast-moving elements. The Gaussian kernel enforces the same symmetric profile across all dimensions, whereas a Beta kernel can provide per-dimension shape control.

## Method

### N-Dimensional Beta Kernel

The core density function is defined as:

$$\sigma(\mathbf{x}, \mathbf{q}) = \mathcal{B}(\mathbf{x}, \mathbf{q}; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \mathbf{b}) \cdot o$$

where $\mathbf{x} \in \mathbb{R}^3$ denotes spatial coordinates, $\mathbf{q} \in \mathbb{R}^{N-3}$ encodes additional dimensions (view direction / time), and $\mathbf{b} \in \mathbb{R}^{N-2}$ controls the Beta shape parameter for each dimension. The Beta exponent for each dimension is $\beta_i = 4\exp(b_i)$:
- **Negative $b_i$**: flat profile (suitable for smooth surfaces, static elements, diffuse reflection)
- **Positive $b_i$**: sharp peak (suitable for fine-grained textures, fast motion, specular reflection)

### Spatial-Orthogonal Cholesky Parameterization

The covariance matrix is decomposed as:

$$\mathbf{L} = \begin{pmatrix} \mathbf{R}_x \text{diag}(\mathbf{s}_x) & \mathbf{0} \\ \mathbf{L}_{qx} & \mathbf{L}_q \end{pmatrix}$$

- $\mathbf{R}_x \in SO(3)$: preserves spatial orthogonal structure (first-order Taylor approximation)
- $\mathbf{L}_{qx}$: encodes cross-dimensional correlations
- Guarantees backward compatibility with the rotation-scale parameterization of 3DGS

### Beta-Modulated Conditional Slice

**Conditional mean and covariance**:

$$\boldsymbol{\mu}_{x|q} = \boldsymbol{\mu}_x + \boldsymbol{\Sigma}_{xq} \boldsymbol{\Sigma}_q^{-1} \text{diag}(\tilde{\boldsymbol{\beta}}_q) (\mathbf{q} - \boldsymbol{\mu}_q)$$

$$\boldsymbol{\Sigma}_{x|q} = \boldsymbol{\Sigma}_x - \boldsymbol{\Sigma}_{xq} \boldsymbol{\Sigma}_q^{-1} \text{diag}(\tilde{\boldsymbol{\beta}}_q) \boldsymbol{\Sigma}_{qx}$$

where $\text{diag}(\tilde{\boldsymbol{\beta}}_q)$ applies Beta modulation to non-spatial dimensions.

**Beta-modulated opacity**:

$$o(\mathbf{q}) = o \prod_{i=1}^C (1 - d_i)^{4\beta_{q_i}}$$

Here $d_i = \tanh(d_i^{raw}) \in [0,1)$ maps the per-dimension Mahalanobis distance to a bounded value.

### Universal Compatibility

| $\mathbf{b}$ Setting | Equivalent Method |
|---------|---------|
| $N=3$, $b_x=0$ | ≈ 3DGS |
| $N=3$, $b_x \neq 0$ | ≈ DBS |
| $N=6$, $\mathbf{b}=\mathbf{0}$ | ≈ 6DGS |
| $N=7$, $\mathbf{b}=\mathbf{0}$ | ≈ 7DGS |

### Interpretable Scene Decomposition

The learned Beta parameters naturally enable unsupervised scene decomposition:
- **Spatial $b_x$**: negative → smooth surfaces; positive → fine-grained textures
- **Angular $b_d$**: negative → diffuse; positive → specular
- **Temporal $b_t$**: negative → static elements; positive → dynamic elements

### Loss & Training

$$\mathcal{L} = (1-\lambda_{SSIM})\mathcal{L}_1 + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_o \sum_i |o_i| + \lambda_\Sigma \sum_i \|\mathbf{s}_i\|_1$$

Opacity regularization ensures effective MCMC densification, while the scale penalty encourages primitive relocation.

### Parameter Efficiency

- Static scenes: **41%** fewer parameters than 3DGS (no 48-parameter spherical harmonics required)
- Dynamic scenes: **73%** fewer parameters than 4DGS

## Key Experimental Results

### Static Scenes

**NeRF Synthetic** (UBS-6D vs. 3DGS vs. 6DGS):

| Scene | 3DGS PSNR | 6DGS PSNR | UBS-6D PSNR |
|------|-----------|-----------|-------------|
| chair | 35.60 | 35.55 | **36.72** |
| ficus | 35.49 | 34.62 | **36.90** |
| materials | 30.50 | 30.63 | **32.90** |
| lego | 36.06 | 35.22 | **36.95** |

PSNR improvements reach up to **+8.27 dB** on the 6DGS-PBR dataset.

### Dynamic Scenes

**D-NeRF and 7DGS-PBR** (UBS-7D vs. 4DGS vs. 7DGS):
- PSNR improvements reach up to **+2.78 dB**
- Particularly pronounced advantages on complex spatio-temporal-angular correlation scenes (cardiac motion, daylight variation, translucent deformation)

### Key Findings

1. **Initializing Beta parameters to zero** guarantees convergence starting from the Gaussian limit.
2. The first-order approximation of spatial-orthogonal Cholesky achieves accuracy comparable to exact rotation, with lower computational cost.
3. The MCMC optimization strategy remains effective for Beta kernels.
4. Real-time rendering performance is on par with 3DGS.

### Training Efficiency

- 30K iterations
- Static: single RTX 4090, approximately 8–10 minutes
- Dynamic: single V100, consistent with 4DGS/7DGS baselines

## Highlights & Insights

1. **Unified framework**: A single Beta kernel primitive jointly handles spatial, angular, and temporal dimensions, replacing multiple specialized encodings.
2. **Backward compatibility**: Setting Beta=0 degrades to the Gaussian case, providing a guaranteed performance lower bound.
3. **Unsupervised decomposition**: Learned Beta parameters naturally disentangle geometry, appearance, and motion.
4. **Substantial parameter reduction**: 41–73% fewer parameters with simultaneous performance gains.
5. **Real-time rendering**: Full CUDA-accelerated implementation.

## Limitations & Future Work

1. The number of parameters in the N-dimensional Cholesky parameterization grows with dimensionality.
2. The bounded support of the Beta kernel may require more primitives in far-field regions.
3. Validation is currently limited to 7 dimensions; effectiveness at higher dimensionalities remains to be investigated.
4. Dynamic scene training requires a batch size of 4, resulting in relatively high GPU memory consumption.
5. The first-order rotation approximation may be insufficiently accurate under extreme rotation angles.

## Related Work & Insights

- **Alternative kernel designs**: GES, TNT-GS, Half-GS, and Disc-GS modify the 3D Gaussian; DBS introduces a 3D Beta kernel.
- **High-dimensional Gaussians**: 6DGS (spatial + view) and 7DGS (spatial + view + time) model conditional distributions over extended dimensions.
- **Dynamic scenes**: D-NeRF employs deformation fields; 4DGS directly extends the temporal dimension.
- **Alternative primitives**: 3D Convex Splatting, Triangle Splatting, and others.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The N-dimensional Beta kernel unified framework constitutes an elegant theoretical contribution.
- **Practicality**: ⭐⭐⭐⭐⭐ — Plug-and-play compatibility combined with real-time rendering.
- **Clarity**: ⭐⭐⭐⭐ — Mathematical derivations are clear, though notation is dense.
- **Significance**: ⭐⭐⭐⭐⭐ — Establishes a universal primitive framework for radiance field rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](../../ICCV2025/3d_vision/splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](../../ICCV2025/3d_vision/stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
