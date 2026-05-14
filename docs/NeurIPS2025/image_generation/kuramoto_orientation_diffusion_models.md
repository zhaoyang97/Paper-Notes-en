---
title: >-
  [Paper Note] Kuramoto Orientation Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Kuramoto model] This work introduces Kuramoto synchronization dynamics from biological systems into score-based generative models…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Kuramoto model"
  - "synchronization dynamics"
  - "orientation field"
  - "periodic domain"
  - "score-based generative models"
date: 2026-05-08
content_hash: 98d763f55ead1323
---

# Kuramoto Orientation Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.15328](https://arxiv.org/abs/2509.15328)
**Code**: [GitHub](https://github.com/KingJamesSong/OrientationDiffusion)
**Area**: Diffusion Models / Image Generation
**Keywords**: Kuramoto model, synchronization dynamics, orientation field, periodic domain, score-based generative models

## TL;DR

This work introduces Kuramoto synchronization dynamics from biological systems into score-based generative models, constructing a forward synchronization / reverse desynchronization diffusion framework over the periodic domain. The proposed approach achieves substantially superior generation quality over standard diffusion models on orientation-dense data such as fingerprints and textures, while remaining competitive on CIFAR-10.

## Background & Motivation

The core structure of orientation-dense images — fingerprints, textures, terrain — is governed by local orientation angles rather than pixel intensity. Such data inherently resides in a periodic domain, and standard diffusion models applying isotropic Euclidean diffusion to angular data suffer from three fundamental issues:

**Periodicity neglect**: Conventional diffusion treats data as continuous quantities in Euclidean space, ignoring the periodic nature of angles (where $-\pi$ and $\pi$ are identical), leading to boundary artifacts.

**Lack of structure in isotropic noise**: The standard forward process employs isotropic Gaussian noise, which rapidly destroys orientation coherence and is ill-suited for structured orientation patterns.

**Low generation efficiency**: Due to the unstructured noise corruption, more diffusion steps are required to generate high-quality orientation-dense images.

The authors draw inspiration from phase synchronization in biological neural systems — the Kuramoto model describes how coupled oscillators spontaneously develop global coherence. This synchronization behavior serves as an inductive bias for structured image generation: local orientations mutually reinforce each other, yielding aligned edges, coherent ridges, and smooth flow fields.

## Method

### Overall Architecture

Pixels are mapped to angular phase variables $\theta_t^i \in [-\pi, \pi]$, and a forward–reverse diffusion process is constructed based on stochastic Kuramoto dynamics. The forward process progressively compresses data into a low-entropy von Mises distribution via synchronization; the reverse process performs desynchronization via a learned score function to generate diverse patterns from the synchronized state.

### Key Designs

1. **Stochastic Kuramoto Forward Process**

   The core SDE is:

   $\frac{d\theta_t^i}{dt} = \frac{1}{N}\sum_{j=1}^{N}K(t)\sin(\theta_t^j - \theta_t^i) + K_{\text{ref}}(t)\sin(\psi_{\text{ref}} - \theta_t^i) + \sqrt{2D_t}\xi^i$

   The three dynamical terms serve distinct roles: (a) Kuramoto sinusoidal coupling between oscillators draws similar phases together; (b) attraction toward a global reference phase $\psi_{\text{ref}}$ ensures final convergence direction; (c) stochastic noise $\sqrt{2D_t}\xi^i$ injects destructive perturbations. The relationship $K_{\text{ref}}(t) > D_t > K(t)$ is maintained to balance structure and noise.

   Under quasi-equilibrium, the terminal distribution is approximated by a von Mises distribution (the circular analogue of the Gaussian):

   $p_{\text{st}}(\theta) \approx \frac{1}{Z}\exp\left(\frac{K(T)r(T)+K_{\text{ref}}(T)}{D_T}\cos(\psi_{\text{ref}}-\theta)\right)$

2. **Local Coupling Variant**

   Global coupling requires each oscillator to interact with all others, which is computationally expensive and inconsistent with the spatial locality of image data. The local coupling variant restricts interactions to a neighborhood $\mathcal{N}_i$:

   $\frac{d\theta_t^i}{dt} = \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i}K(t)\sin(\theta_t^j - \theta_t^i) + K_{\text{ref}}(t)\sin(\psi_{\text{ref}} - \theta_t^i) + \sqrt{2D_t}\xi^i$

   Local coupling introduces spatial inhomogeneity and produces a blurring effect analogous to heat diffusion, better reflecting the spatial correlations inherent in image data.

3. **Wrapped Gaussian Transition Kernel and Periodicity-Aware Network**

   Due to phase wrapping, local transition probabilities follow a Wrapped Gaussian distribution (approximated via truncated summation with $K=3$ terms). The score network takes sinusoidal embeddings $[\sin(\theta), \cos(\theta)]$ as input and outputs two Cartesian components $s_1, s_2$, with angular-domain projection ensuring periodic consistency:

   $s(\theta, t) = s_1(\theta, t)\cos(\theta) + s_2(\theta, t)\sin(\theta)$

### Loss & Training

Training is based on Local Score Matching, using Monte Carlo sampling from the forward transition kernel to estimate the loss:

$$\mathcal{L} = \frac{1}{M}\sum_{m=0}^{M-1}\left(2D_t\|s(\theta_t^m, t) - \nabla_{\theta_t^m}\log p(\theta_t^m|\theta_{t-1})\|^2\right)$$

At each step, the forward Markov chain is simulated to obtain $\theta_{t-1}$, and $M=5$ samples of $\theta_t$ are drawn from the local transition kernel. Pixel values are mapped from $[-1,1]$ to $[-0.9\pi, 0.9\pi]$, with boundary margins reserved to avoid aliasing from phase wrapping.

## Key Experimental Results

### Main Results

| Dataset | Steps | SGM | Kuramoto (Global) | Kuramoto (Local) | Local Gain |
|---------|-------|-----|-------------------|------------------|------------|
| SOCOFing Fingerprint | 100 | 104.92 | 74.41 | **67.49** | −35.7% |
| SOCOFing Fingerprint | 1000 | 23.84 | 20.64 | **18.75** | −21.4% |
| Brodatz Texture | 100 | 38.33 | 20.26 | **18.47** | −51.8% |
| Brodatz Texture | 1000 | 20.37 | 15.42 | **14.19** | −30.3% |
| Terrain | 100 | 114.90 | 101.65 | **92.86** | −19.2% |
| Terrain | 1000 | 33.79 | 33.56 | **30.62** | −9.4% |

### CIFAR-10 Comparison

| Steps | SGM | Kuramoto (Global) | Kuramoto (Local) |
|-------|-----|-------------------|------------------|
| 100 | 38.04 | 29.96 | **28.17** |
| 300 | **25.76** | 25.83 | 24.86 |
| 1000 | **3.17** | 11.58 | 10.79 |

### Key Findings

1. **Large advantage on orientation-dense data**: On Brodatz textures at 100 steps, the Kuramoto model achieves an FID nearly 52% lower than SGM, and 100-step Kuramoto performance approaches or surpasses SGM at 1000 steps.
2. **Clear advantage at low step counts**: The synchronizing forward process enables faster convergence to the terminal distribution, allowing the reverse process to generate high-quality samples with fewer steps.
3. **Trade-off on CIFAR-10**: At low-step configurations, Kuramoto substantially outperforms SGM (FID 28.17 vs. 38.04 at 100 steps), but SGM is superior at 1000 steps (3.17 vs. 10.79), indicating that the synchronization bias slightly limits expressiveness for natural images lacking strong orientation priors when given sufficient steps.
4. **Hierarchical generation**: The reverse process exhibits coarse-to-fine hierarchical generation — global structure is established first, followed by progressive refinement of details.

## Highlights & Insights

1. **Deep biological inspiration**: Rather than a superficial analogy, Kuramoto synchronization dynamics are rigorously embedded within the SDE framework of diffusion models in a mathematically consistent manner.
2. **Value of non-isotropic diffusion**: While isotropic Gaussian noise is conventionally regarded as the default for diffusion models, this work demonstrates that structured non-isotropic noise yields significant advantages in specific domains.
3. **Clear mechanism linking synchronization to FID gains**: The synchronization bias preserves global structure early in the process, providing the reverse process with a better starting point — directly explaining the FID advantage at low step counts.

## Limitations & Future Work

- Training incurs $\mathcal{O}(T)$ forward chain simulation cost per step (mitigable via precomputed caching).
- Performance on natural images (lacking orientation priors) falls below SGM at 1000 steps; the structural bias may limit long-range flexibility.
- Validation is currently limited to resolutions from $32\times32$ to $128\times128$; scalability to high resolutions remains to be examined.
- The selection of local coupling neighborhood size lacks an adaptive mechanism.

## Related Work & Insights

- **Geometry-aware diffusion models**: Generative models on non-Euclidean manifolds, including Riemannian Flow Matching and hyperspherical VAEs.
- **Neural oscillations**: The AKOrN framework replaces threshold activation functions with Kuramoto oscillators.
- **Structured diffusion**: Blurring diffusion models (Rissanen et al.) employ the heat equation as the forward process.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First work to introduce Kuramoto synchronization dynamics into generative models; theoretical construction is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ — Multi-dataset, multi-step comparisons are thorough, but high-resolution and large-scale experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, theoretical derivations are complete, and visualizations are rich.
- Value: ⭐⭐⭐⭐☆ — Opens a new direction for nonlinear dynamics-driven generative models; high application value for orientation-dense data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Orient Anything V2: Unifying Orientation and Rotation Understanding](orient_anything_v2_unifying_orientation_and_rotation_understanding.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] System-Embedded Diffusion Bridge Models](system-embedded_diffusion_bridge_models.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] Diffusion Models Meet Contextual Bandits](diffusion_models_meet_contextual_bandits.md)

</div>

<!-- RELATED:END -->
