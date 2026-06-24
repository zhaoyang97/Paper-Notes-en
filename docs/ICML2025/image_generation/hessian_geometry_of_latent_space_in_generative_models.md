---
title: >-
  [Paper Note] Hessian Geometry of Latent Space in Generative Models
description: >-
  [ICML2025][Image Generation][Fisher Information Metric] A method is proposed to analyze the latent space geometry of generative models by reconstructing the Fisher information metric, revealing that fractal-structured phase transition boundaries exist within the latent space of diffusion models, where the Lipschitz constant diverges.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Fisher Information Metric"
  - "Hessian Geometry"
  - "Latent Space Analysis"
  - "Phase Transition"
  - "Diffusion Models"
  - "Geodesics"
date: 2026-05-08
content_hash: dd248102273b904e
---

# Hessian Geometry of Latent Space in Generative Models

**Conference**: ICML2025  
**arXiv**: [2506.10632](https://arxiv.org/abs/2506.10632)  
**Code**: [GitHub](https://github.com/alobashev/hessian-geometry-of-diffusion-models)  
**Area**: Generative Model Theory / Information Geometry  
**Keywords**: Fisher Information Metric, Hessian Geometry, Latent Space Analysis, Phase Transition, Diffusion Models, Geodesics

## TL;DR

A method is proposed to analyze the latent space geometry of generative models by reconstructing the Fisher information metric, revealing that fractal-structured phase transition boundaries exist within the latent space of diffusion models, where the Lipschitz constant diverges.

## Background & Motivation

- **Core Problem**: Image generative models often exhibit abrupt transitions (such as sudden changes in content) during latent space interpolation, indicating that the latent space is not smooth, yet existing methods lack systematic tools for geometric analysis.
- **Two Research Lines**:
    1. **Riemannian Geometry of Latent Space**: Park et al. (2023) constructed latent space bases using Jacobian singular vectors, and Shao et al. (2018) computed geodesics using pullback metrics in pixel space, but these are only applicable to deterministic generators.
    2. **Learning Phase Transitions in Statistical Physics**: ML methods have been used to identify phase transition boundaries in systems like the Ising model; Walker et al. (2020) found that VAEs implicitly extract sufficient statistics.
- **Motivation**: To treat generative models uniformly as statistical physical systems and analyze their latent space structure using information geometry methods, which can be applied to stochastic generation processes (such as stochastic sampling in diffusion models).

## Method

### Overall Architecture: Two-Step Reconstruction of the Fisher Metric

**Step 1: Posterior Distribution Approximation**

Given a generative model $p(x|t)$, approximate the posterior distribution $p(t|x_1, \dots, x_N)$ from samples $x_1, \dots, x_N \sim p(x|t')$.

**Key Theorem (Theorem 3.1)**: For exponential family distributions, as $N \to \infty$:

$$\lim_{N \to \infty} \left(p(t|x_1, \dots, x_N)\right)^{1/N} = e^{-D_{\log Z(t)}(t, t')}$$

where $D_{\log Z(t)}(t, t')$ is the Bregman divergence, which is equivalent to the KL divergence $D_{\text{KL}}(p(x|t') \| p(x|t))$ for the exponential family.

**Two Posterior Approximation Strategies**:

- **Directly Training Mapping** (U²-Net): Applicable to statistical physics models (Ising, TASEP), where samples are stochastic and pixel-wise uncorrelated.
- **Feature Extractor** (CLIP): Applicable to the image domain, reconstructing via $D_{\text{KL}} \approx \frac{1}{2}\|\mathcal{E}(x_1) - \mathcal{E}(x_2)\|^2$.

**Step 2: Fisher Metric Reconstruction**

**Key Theorem (Theorem 3.2)**: Minimizing the MSE loss of the Bregman divergence recovers the Hessian of $\log Z(t)$ (i.e., the Fisher metric):

$$g_F(t) = \nabla^2 \log Z(t)$$

In practice, the Jensen-Shannon divergence is used instead of MSE during training to avoid vanishing gradients:

$$\mathcal{L}_1(\theta) = \int_{\mathcal{S}} D_{\text{JS}}\left(p(t|x_1, \dots, x_N),\; p_{\log Z_\theta}(t|t')\right) dt'$$

- Parameterize $\log Z_\theta(t)$ using an MLP with 5 hidden layers, 512 dimensions, and ReLU activations.
- The MLP is not required to satisfy convexity constraints; it naturally converges to a convex function during training.

### Geodesic Approximation

After obtaining the Fisher metric, the curve $\gamma(t)$ is discretized as $\{\gamma_0, \gamma_1, \dots, \gamma_N\}$, and the path length is minimized using Adam to optimize the intermediate points:

$$L[\gamma(t)] = \int_0^1 \sqrt{\dot{\gamma}(t)^T g_F(\gamma(t)) \dot{\gamma}(t)} \, dt$$

### Theoretical Foundation

- **Bryant–Amari–Armstrong Theorem**: Any 2D analytic Riemannian metric can be locally represented as a Hessian structure, ensuring the method's theoretical validity for arbitrary 2D latent space slices.
- **Exponential Family Properties**: The Fisher metric equals the Hessian of the log partition function, and $\log Z(t)$ only needs to be reconstructed up to an affine transformation.

## Key Experimental Results

### Ising Model & TASEP: Exactly Solvable Verification

| Model | Method | F RMSE | dF/d(param1) RMSE | dF/d(param2) RMSE |
|------|------|--------|-------------------|-------------------|
| **Ising** | Convex (Ours) | **0.0883±0.0006** | **0.1106±0.0002** | **0.1237±0.0016** |
| Ising | Mean-as-Stat | 0.0981±0.0010 | 0.4766±0.0023 | 1.0936±0.0033 |
| Ising | PCA-VAE | 0.1669±0.0018 | 0.7428±0.0025 | 0.7988±0.0022 |
| **TASEP** | Convex (Ours) | **0.0112±0.00008** | **0.1165±0.0025** | **0.1135±0.0017** |
| TASEP | Mean-as-Stat | 0.0529±0.0005 | 0.3832±0.0038 | 0.3833±0.0031 |
| TASEP | PCA-VAE | 0.0524±0.0006 | 0.3837±0.0038 | 0.3872±0.0022 |

The reconstruction accuracy of partial derivatives is improved by 3–8×, which is crucial for identifying phase transition boundaries.

### Diffusion Models: 2D Latent Space Slice Analysis

Experiments are based on StableDiffusion 1.5 (Dreamshaper8) with a DDIM scheduler (50 steps, CFG=5), generating 60,000 images per group.

| Metric | Geodesic (Ours) | Linear | Geodesic (Wang/Shao) |
|------|----------------|--------|---------------------|
| CLIP Length | **72.3±4.00** | 73.6±3.54 | 73.6±4.37 |
| Pixel Length | 2.77×10⁶ | 2.76×10⁶ | **2.74×10⁶** |
| PPL | **3.12±0.16** | 3.17±0.23 | 3.19±0.21 |

**Key Findings**:

- The reconstructed $\log Z(\alpha, \beta)$ is non-smooth, with abrupt changes in derivatives $\to$ reflecting phase transitions in the image space.
- **Fractal Phase Boundary**: Zooming in on the phase boundary reveals self-similar structures, spanning scales from $10^{-5}$ down to the float16 precision limit of $10^{-8}$.
- **Geodesics within a single phase are approximately linear**, but this linearity breaks down at the phase boundaries.
- The Lipschitz constant of diffusion models with respect to the latent space diverges at the phase boundary (this is a novel finding to the best of the authors' knowledge).

## Highlights & Insights

1. **Unified Framework**: Statistics physics and generative models are unified via information geometry. The method is applicable to both exponential family systems (Ising, TASEP) and non-exponential family ones (2D slices of diffusion models, guaranteed by the Bryant–Amari–Armstrong theorem).
2. **Discovery of Fractal Phase Transitions**: The phase boundaries in the latent space of diffusion models exhibit fractal structures, which is a fundamental difference from classical continuous phase boundaries.
3. **Lipschitz Divergence**: This work is the first to report the phenomenon of the Lipschitz constant of diffusion models diverging with respect to the latent space.
4. **Practical Value**: Geodesic interpolation guided by the Fisher metric is perceptually smoother than linear interpolation.
5. **JSD Training Trick**: Using the Jensen-Shannon divergence instead of the MSE loss addresses the vanishing gradient problem.

## Limitations & Future Work

1. **Dimension Limitations**: Currently, the method is restricted to 2D latent space slices (relying on the Bryant–Amari–Armstrong theorem), and a full analysis of higher-dimensional latent spaces remains unfeasible.
2. **High Computational Overhead**: Generating 60,000 images is required for each 2D slice, limiting general scalability.
3. **Limitations of CLIP Approximation**: Using the CLIP distance as an approximation of the KL divergence relies on normality assumptions, which may not hold in practice.
4. **Deterministic vs. Stochastic Sampling**: When the DDIM parameter $\eta > 0$, the phase boundaries in the CLIP scheme are blurred, making the method sensitive to noise.
5. **Limited Geodesic Performance Gain**: The performance difference between methods in Table 2 is minor; the practical advantages of geodesics are primarily manifested in cross-phase boundary scenarios.
6. **Only Evaluated on SD1.5**: The method has not been verified on more advanced diffusion models (such as SDXL, SD3).

## Related Work & Insights

- Park et al. (2023): Latent space base construction using Jacobian singular vectors
- Shao et al. (2018): Pullback metric geodesics (limited to deterministic models)
- Walker et al. (2020): VAEs implicitly extracting sufficient statistics of the Ising model
- Wang et al. (2021): LPIPS pullback metric for GAN latent space
- Yang et al. (2023): Discussions on Lipschitz constants with respect to the time variable in diffusion models

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (A triple intersection of information geometry, statistical physics, and generative models; the discovery of fractal phase transitions is highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Verification on exactly solvable models is thorough, but diffusion model experiments are limited to SD1.5 and 2D slices)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear, though the LaTeX notation is dense in places)
- Value: ⭐⭐⭐⭐ (Provides a new perspective for understanding the latent space of generative models, but its practicality is limited by dimensionality)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)
- [\[CVPR 2025\] Latent Space Imaging](../../CVPR2025/image_generation/latent_space_imaging.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](../../ICCV2025/image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[ICML 2025\] Reimagining Parameter Space Exploration with Diffusion Models](reimagining_parameter_space_exploration_with_diffusion_models.md)
- [\[CVPR 2025\] Probability Density Geodesics in Image Diffusion Latent Space](../../CVPR2025/image_generation/probability_density_geodesics_in_image_diffusion_latent_space.md)

</div>

<!-- RELATED:END -->
