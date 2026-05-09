---
title: >-
  [Paper Note] The Spacetime of Diffusion Models: An Information Geometry Perspective
description: >-
  [ICLR 2026][Image Generation][Spacetime Geometry] This paper proposes a "spacetime" framework for diffusion models from an information-geometric perspective. It proves that the standard pullback geometry degenerates to straight lines in diffusion models, and introduces instead a spacetime geometry based on the Fisher-Rao metric, from which practically computable diffusion edit distances (DiffED) and transition path sampling methods are derived.
tags:
  - ICLR 2026
  - Image Generation
  - Spacetime Geometry
  - Fisher-Rao Metric
  - Pullback Geometry
  - Diffusion Edit Distance
  - Transition Path Sampling
date: 2026-05-08
content_hash: b850e72cd9eac839
---

# The Spacetime of Diffusion Models: An Information Geometry Perspective

**Conference**: ICLR 2026
**arXiv**: [2505.17517](https://arxiv.org/abs/2505.17517)
**Code**: [GitHub](https://github.com/rafalkarczewski/spacetime-geometry)
**Area**: Diffusion Models / Information Geometry / Theoretical Analysis
**Keywords**: Spacetime Geometry, Fisher-Rao Metric, Pullback Geometry, Diffusion Edit Distance, Transition Path Sampling

## TL;DR

This paper proposes a "spacetime" framework for diffusion models from an information-geometric perspective. It proves that the standard pullback geometry degenerates to straight lines in diffusion models, and introduces instead a spacetime geometry based on the Fisher-Rao metric, from which practically computable diffusion edit distances (DiffED) and transition path sampling methods are derived.

## Background & Motivation

Understanding the information evolution of intermediate noisy states $\mathbf{x}_t$ in diffusion models remains an open problem:

**Failure of pullback geometry**: In generative models, the intrinsic geometry of data is typically studied via pullback of the ambient metric. However, this approach suffers from a fundamental issue in diffusion models.

**Lack of understanding of intermediate-state geometry**: Existing work focuses primarily on sampling and training, with little analysis of how information evolves through the noising process.

**Need for principled notions of distance and path**: Existing image similarity metrics (e.g., LPIPS) lack a geometric foundation grounded in the generative process.

## Method

### 1. Degeneration of Pullback Geometry (Core Negative Result)

**Theorem**: The pullback metric of the deterministic PF-ODE decoder $\mathbf{x}_T \mapsto \mathbf{x}_0(\mathbf{x}_T)$,

$$\mathbf{G}_{\text{PB}}(\mathbf{x}_T) = \left(\frac{\partial \mathbf{x}_0}{\partial \mathbf{x}_T}\right)^\top \left(\frac{\partial \mathbf{x}_0}{\partial \mathbf{x}_T}\right)$$

causes all geodesics to decode as **straight line segments** in data space.

**Reason**: In diffusion models, the latent and data spaces share the same dimensionality; the decoder operates in the ambient space and is thus unable to capture the intrinsic structure of the data manifold.

### 2. The Memorylessness Problem in Information Geometry

The Fisher-Rao metric of the stochastic decoder (reverse SDE) is:

$$\mathbf{G}_{\text{IG}}(\mathbf{x}_T) = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|\mathbf{x}_T)}[\nabla_{\mathbf{x}_T}\log p(\mathbf{x}_0|\mathbf{x}_T) \nabla_{\mathbf{x}_T}\log p(\mathbf{x}_0|\mathbf{x}_T)^\top]$$

However, due to memorylessness: $p(\mathbf{x}_T|\mathbf{x}_0) \approx p_T(\mathbf{x}_T)$, the Fisher-Rao metric collapses to zero at $\mathbf{x}_T$.

### 3. Latent Spacetime

**Core Innovation**: A $(D+1)$-dimensional spacetime $\mathbf{z} = (\mathbf{x}_t, t) \in \mathbb{R}^D \times (0, T]$ is introduced to:

- Index the family of denoising distributions $\{p(\mathbf{x}_0|\mathbf{x}_t)\}$ across all noise levels
- Recover a non-degenerate geometric structure
- Identify clean data as spacetime points $(\mathbf{x}, 0)$

### 4. Exponential Family Structure and Computable Energy

**Proposition**: The denoising distributions form an exponential family, and the spacetime curve energy admits a closed-form approximation:

$$\mathcal{E}(\boldsymbol{\gamma}) \approx \frac{N-1}{2}\sum_{n=0}^{N-2}(\boldsymbol{\eta}(\mathbf{z}_{n+1}) - \boldsymbol{\eta}(\mathbf{z}_n))^\top(\boldsymbol{\mu}(\mathbf{z}_{n+1}) - \boldsymbol{\mu}(\mathbf{z}_n))$$

where the natural and expectation parameters are:

$$\boldsymbol{\eta}(\mathbf{x}_t, t) = \left(\frac{\alpha_t}{\sigma_t^2}\mathbf{x}_t, -\frac{\alpha_t^2}{2\sigma_t^2}\right)$$

$$\boldsymbol{\mu}(\mathbf{x}_t, t) = \left(\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t], \mathbb{E}[\|\mathbf{x}_0\|^2|\mathbf{x}_t]\right)$$

**Computation**: Via the Tweedie formula and Hutchinson's trick, estimation requires only a single Jacobian-vector product (JVP).

### 5. Diffusion Edit Distance (DiffED)

$$\text{DiffED}(\mathbf{x}^a, \mathbf{x}^b) = \ell(\boldsymbol{\gamma})$$

where $\boldsymbol{\gamma}$ is the spacetime geodesic connecting $(\mathbf{x}^a, 0)$ and $(\mathbf{x}^b, 0)$.

**Intuition**: The geodesic traces the minimal edit sequence — adding sufficient noise to forget the information unique to $\mathbf{x}^a$, then denoising to introduce the information unique to $\mathbf{x}^b$. The distance measures the total change in the denoising distribution along the path.

### 6. Transition Path Sampling

For a Boltzmann distribution $q(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$:
- Estimate the spacetime geodesic between two low-energy states
- Sample along the geodesic using annealed Langevin dynamics
- Supports constrained variants (low-variance paths, region avoidance)

## Key Experimental Results

### Sampling Trajectory Comparison
- PF-ODE paths closely resemble energy-minimizing geodesics
- Geodesics curve slightly less during the early sampling phase

### Diffusion Edit Distance

| Property | Result |
|----------|--------|
| Correlation with LPIPS | ~−7% (captures different information) |
| Correlation with SSIM | ~53% |
| Less similar endpoints | Stronger intermediate noise |

DiffED captures structural edit cost rather than perceptual similarity.

### Transition Path Sampling (Alanine Dipeptide)

| Method | MaxEnergy↓ | Energy Evaluations↓ |
|--------|-----------|---------------------|
| MCMC-Fixed Length | 42.54±7.42 | 1.29B |
| MCMC-Variable Length | 58.11±18.51 | 21.02M |
| Doob's Lagrangian | 66.24±1.01 | 38.4M |
| **Spacetime Geodesic (Ours)** | **37.36±0.60** | **16M (+16M)** |
| Lower Bound | 36.42 | — |

The proposed method most closely approaches the lower bound while requiring orders of magnitude fewer energy evaluations.

### Constrained Paths
- Generated paths effectively avoid high-energy regions
- Unlike Doob's Lagrangian, paths do not collapse to a single trajectory

## Highlights & Insights

1. **Deep theoretical insight**: Formally proves the fundamental failure of pullback geometry in diffusion models
2. **Elegance of the spacetime concept**: Unifies the geometric structure across all noise levels
3. **Computability**: Derives simulation-free estimators by exploiting the exponential family structure
4. **Multi-domain applicability**: Edit distance + molecular dynamics
5. **Computational efficiency**: Energy estimation requires only a single JVP

## Limitations & Future Work

1. Spacetime geodesics cannot serve as an alternative sampling method, as both endpoints must be known in advance
2. The Hutchinson estimator may introduce variance in high-dimensional settings
3. The computational cost of DiffED remains higher than that of simple distance metrics
4. Results depend on the quality of the denoiser (approximation error in $\hat{\mathbf{x}}_0$)
5. Transition path sampling requires a known energy function

## Related Work & Insights

- **Riemannian geometry + generative models**: Arvanitidis (2018/2022), Park (2023)
- **Geometry of diffusion models**: Domingo-Enrich (2025), memorylessness analysis
- **Transition path sampling**: Holdijk (2023), Doob's Lagrangian (Du 2024)
- **Information geometry**: Fisher-Rao metric, Amari (2016)

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The spacetime geometry concept is highly original and intellectually deep
- **Utility**: ⭐⭐⭐⭐ — DiffED and transition path sampling offer practical value
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Theoretical validation is thorough; molecular dynamics results are strong
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretically elegant with precise exposition

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICLR 2026\] When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis](when_scores_learn_geometry_rate_separations_under_the_manifold_hypothesis.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](../../NeurIPS2025/image_generation/information_theoretic_learning_for_diffusion_models_with_warm_start.md)

</div>

<!-- RELATED:END -->
