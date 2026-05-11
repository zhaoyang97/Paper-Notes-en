---
title: >-
  [Paper Note] Guided Diffusion Sampling on Function Spaces with Applications to PDEs
description: >-
  [NeurIPS 2025][Image Generation][function-space diffusion models] This paper proposes **FunDPS (Function-space Diffusion Posterior Sampling)**, which trains an unconditional diffusion model in function space and performs…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "function-space diffusion models"
  - "PDE inverse problems"
  - "posterior sampling"
  - "Tweedie formula"
  - "neural operators"
  - "resolution invariance"
date: 2026-05-08
content_hash: 5ca96608985b6b6d
---

# Guided Diffusion Sampling on Function Spaces with Applications to PDEs

**Conference**: NeurIPS 2025
**arXiv**: [2505.17004](https://arxiv.org/abs/2505.17004)
**Code**: [neuraloperator/FunDPS](https://github.com/neuraloperator/FunDPS)
**Area**: Image Generation
**Keywords**: function-space diffusion models, PDE inverse problems, posterior sampling, Tweedie formula, neural operators, resolution invariance

## TL;DR

This paper proposes **FunDPS (Function-space Diffusion Posterior Sampling)**, which trains an unconditional diffusion model in function space and performs plug-and-play posterior sampling for PDE inverse problems via gradient guidance at inference time. Theoretically, it extends the Tweedie formula to infinite-dimensional Banach spaces. Empirically, across 5 PDE tasks with only 3% observations, FunDPS achieves 32% higher accuracy on average than DiffusionPDE while reducing the number of sampling steps by 4×.

## Background & Motivation

### Problem Setting

Many tasks in scientific computing reduce to **conditional sampling / inverse problems**: recovering complete physical fields (temperature, flow, permeability, etc.) from sparse or noisy measurements. Representative examples include:

- Subsurface flow: recovering Darcy flow permeability fields from sparse sensor readings
- Climate prediction: inferring global atmospheric states from limited station observations
- Elasticity: identifying material properties from displacement fields

### Limitations of Prior Work

**MCMC methods**: Theoretically sound but prohibitively slow to converge in high dimensions; constructing effective proposal distributions is non-trivial.

**Deterministic neural PDE solvers** (FNO, DeepONet, PINN): Produce only point estimates without posterior distributions; errors become severe under extremely sparse observations.

**Finite-dimensional diffusion models** (DiffusionPDE): Modeled in fixed-resolution pixel space, requiring retraining when the resolution changes; require up to 2000 sampling steps, resulting in slow inference.

**Function-space diffusion models** (DDO, etc.): Existing models support only unconditional generation, or require training a dedicated conditional score model for each observation configuration, limiting flexibility.

### Core Insight

Physical systems are intrinsically described by continuous functions, so prior distributions should be modeled in **function space**. Training a single unconditional function-space diffusion model and injecting observation constraints at inference time in a plug-and-play manner enables the same model to handle diverse downstream inverse problems—without retraining for different sensor configurations.

## Core Problem

Given extremely sparse measurements $\boldsymbol{u} = \boldsymbol{A}(\boldsymbol{a}) + \varepsilon$ (only 3% of spatial points observed) in a PDE system, how can one sample from the posterior distribution $\nu(\boldsymbol{a}|\boldsymbol{u})$ in function space to achieve resolution-invariant, high-accuracy solutions to both forward and inverse problems?

## Method

### Overall Architecture

FunDPS consists of two stages:

1. **Training**: An unconditional diffusion model is trained in function space to learn the joint prior distribution over PDE parameters and solutions.
2. **Inference**: The posterior score is decomposed via Bayes' rule into a prior score plus a likelihood gradient. The trained denoiser approximates the prior; the Tweedie formula approximates the likelihood, enabling plug-and-play guided sampling.

### Bayesian Framework

Under the function-space setting, a Bayesian perspective is adopted:

- **Prior** $\nu(\boldsymbol{a})$: Learned from data by the diffusion model.
- **Likelihood** $p(\boldsymbol{u}|\boldsymbol{a})$: Determined by the forward operator $\boldsymbol{A}$ and Gaussian measurement noise $\eta = \mathcal{N}(0, \mathbf{C}_\eta)$.
- **Posterior** $\nu^{\boldsymbol{u}}(\boldsymbol{a}) \propto \nu(\boldsymbol{a}) \exp(\Phi(\boldsymbol{a}, \boldsymbol{u}))$: The Radon-Nikodym derivative of the likelihood is obtained via the Cameron-Martin theorem.

The key conditional score decomposition (analogous to finite-dimensional DPS) is:

$$\nabla_{\boldsymbol{a}_t} \log \frac{d\nu_t^{\boldsymbol{u}}}{d\gamma_t}(\boldsymbol{a}_t) = \underbrace{\nabla_{\boldsymbol{a}_t} \log \frac{d\nu_t}{d\gamma_t}(\boldsymbol{a}_t)}_{\text{Prior score (trained)}} + \underbrace{\nabla_{\boldsymbol{a}_t} \tilde{\Phi}_t(\boldsymbol{a}_t, \boldsymbol{u})}_{\text{Likelihood gradient (approximated)}}$$

### Infinite-Dimensional Tweedie Formula (Core Theoretical Contribution)

In finite dimensions, the Tweedie formula provides a closed-form relationship between the conditional expectation $\mathbb{E}[\boldsymbol{a}_0|\boldsymbol{a}_t]$ and the score function, which serves as the theoretical foundation for guided sampling methods such as DPS and MCG. However, this result had previously only been established in finite-dimensional settings.

**Theorem 3.1**: Let $B$ be a separable Banach space. Under appropriate regularity conditions, for $\nu$-almost every $y$:

$$\mathbb{E}[X|Y=y] = R\left(D_{H(\gamma)} \log \frac{d\nu}{d\gamma}(y)\right)$$

where $R$ is the Riesz representation map and $D_{H(\gamma)}$ is the Fréchet derivative along the Cameron-Martin space.

This generalization allows the trained denoiser $\boldsymbol{D}_\theta(\boldsymbol{a}_t, t) \approx \mathbb{E}[\boldsymbol{a}_0|\boldsymbol{a}_t]$ to approximate the likelihood:

$$\nabla_{\boldsymbol{a}_t} \tilde{\Phi}_t(\boldsymbol{a}_t, \boldsymbol{u}) \approx -\frac{c}{2} \nabla_{\boldsymbol{a}_t} \|\boldsymbol{u} - \boldsymbol{A}(\hat{\boldsymbol{a}}_0(\boldsymbol{a}_t))\|_{\mathcal{U}}^2$$

### Guided Update Rule

At each step of the reverse diffusion process, samples are updated as:

$$\boldsymbol{a}_{i+1} \leftarrow \boldsymbol{a}_i - \boldsymbol{\zeta} \cdot \nabla_{\boldsymbol{a}_i} \|\boldsymbol{u} - \boldsymbol{A}(\boldsymbol{D}_\theta(\boldsymbol{a}_i, t_i))\|_{\mathcal{U}}^2$$

where $\boldsymbol{\zeta}$ is a predefined guidance weight vector that collectively absorbs the scaling factor from the noise covariance and the confidence of each observation channel.

### Joint Embedding

The PDE parameter functions (coefficients $c$, boundary conditions $g$) and solution function $f$ are **jointly represented** as a multi-channel function $\boldsymbol{a}$. By applying channel-wise masking:

- **Forward problem**: The solution channel is fully masked; the (sparse) parameter channel is retained.
- **Inverse problem**: The parameter channel is fully masked; the (sparse) solution channel is retained.
- **Mixed problem**: Both channels are partially masked simultaneously.

### Multi-Resolution Training

Exploiting the discretization invariance of neural operators (U-shaped Neural Operator), a curriculum learning strategy is adopted:

1. The majority of training epochs proceed on low-resolution data to learn coarse-grained structure.
2. A small number of final epochs fine-tune on high-resolution data to capture high-frequency details.
3. Total training GPU time is reduced by **25%**, with accuracy on par with full-resolution training throughout.

### Multi-Resolution Inference (ReNoise)

A **ReNoise** two-level multi-resolution inference strategy is proposed:

1. The first 80% of denoising steps are executed at low resolution.
2. The result is upsampled to the target resolution.
3. Additional noise is injected to remove upsampling artifacts and correct noise-level mismatches.
4. The final 20% of steps refine high-frequency details at the target resolution.

This strategy provides an additional **2×** speedup.

### Network Architecture

- Denoiser $\boldsymbol{D}_\theta$: U-shaped Neural Operator (based on EDM-FS), approximately 54M parameters.
- Noise sampler: Gaussian Random Fields (GRF) rather than multivariate Gaussians, ensuring function-space consistency.
- Sampler: Second-order deterministic sampler.

## Key Experimental Results

### Experimental Setup

- **5 PDE tasks**: Darcy Flow, Poisson, Helmholtz, Navier-Stokes (periodic BC), Navier-Stokes (Dirichlet BC).
- **Observation density**: Only 3% of spatial points are observed — an extremely sparse setting.
- **Baselines**: FNO, PINO, DeepONet, PINN, DiffusionPDE.
- **Resolution**: $128 \times 128$.

### Main Results ($L^2$ relative error, %, across 5 PDEs × forward/inverse = 10 subtasks)

| Method | Steps | Darcy Fwd | Darcy Inv | Poisson Fwd | Poisson Inv | Helmholtz Fwd | Helmholtz Inv | NS Fwd | NS Inv | NS-BC Fwd | NS-BC Inv |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **FunDPS** | 200 | 2.88 | 6.78 | 2.04 | 24.04 | 2.20 | 20.07 | 3.99 | 9.87 | 5.91 | 4.31 |
| **FunDPS** | 500 | **2.49** | **5.18** | **1.99** | **20.47** | **2.13** | **17.16** | **3.32** | **8.48** | **4.90** | **4.08** |
| DiffusionPDE | 2000 | 6.07 | 7.87 | 4.88 | 21.10 | 12.64 | 19.07 | 3.78 | 9.63 | 9.69 | 4.18 |
| FNO | - | 28.2 | 49.3 | 100.9 | 232.7 | 98.2 | 218.2 | 101.4 | 96.0 | 82.8 | 69.6 |
| PINN | - | 48.8 | 59.7 | 128.1 | 130.0 | 142.3 | 160.0 | 142.7 | 146.8 | 100.1 | 105.5 |

Key findings:

- FunDPS (500 steps) achieves the best results across all tasks, reducing average error by **32%** compared to DiffusionPDE.
- FunDPS at 200 steps already surpasses DiffusionPDE at 2000 steps.
- Deterministic baselines (FNO, PINN, etc.) fail completely under the 3% sparse observation setting (errors of 30%–200%+).

### Inference Speed Comparison

| Method | Steps | Time per Sample | Hardware |
|---|---|---|---|
| FunDPS | 500 | **15 s** | RTX 4090 |
| FunDPS + ReNoise | 500 | **7.5 s** | RTX 4090 |
| DiffusionPDE | 2000 | 190 s | RTX 4090 |

Combined with ReNoise, FunDPS achieves **25×** faster inference than DiffusionPDE at higher accuracy.

### Multi-Resolution Training

Multi-resolution curriculum training requires only 25% of the GPU time of full-resolution training, with negligible accuracy degradation.

### Multi-Resolution Inference

ReNoise with 80% of steps at low resolution matches the accuracy of full-resolution inference, delivering an additional 2× speedup.

## Highlights & Insights

1. **Strong theoretical contribution**: The Tweedie formula is rigorously extended to infinite-dimensional Banach spaces for the first time, providing a mathematical foundation for posterior sampling in function space.
2. **Plug-and-play flexibility**: A single unconditional model is trained once; arbitrary observation operators are composed at inference time without retraining.
3. **Resolution invariance**: Built on neural operators, the model natively supports cross-resolution inference and can be applied to different mesh densities with the same weights.
4. **Dramatic speedup**: Achieves 25× faster inference than DiffusionPDE while improving accuracy by 32%.
5. **Multi-resolution training/inference**: Curriculum learning reduces training GPU time by 75%; ReNoise inference provides an additional 2× speedup.
6. **Unified framework**: The joint embedding design allows a single model to handle forward, inverse, and mixed problems uniformly.

## Limitations & Future Work

1. **Limited benefit from PDE loss guidance**: Incorporating finite-difference PDE residuals directly into the guidance yields only marginal improvements, likely due to accumulated discretization errors.
2. **Manual tuning of guidance weights**: The weight vector $\boldsymbol{\zeta}$ requires task-specific tuning; an adaptive weighting scheme is lacking.
3. **Inaccurate approximation at high noise levels**: The likelihood approximation $\tilde{\Phi}_t \approx \Phi(\hat{\boldsymbol{a}}_0, \boldsymbol{u})$ degrades in accuracy during the early stages of diffusion (high noise).
4. **Unvalidated on specialized inverse problems**: Performance on domain-specific tasks such as MRI reconstruction and full waveform inversion remains unknown.
5. **Temporal dimension not yet addressed**: The current work focuses on spatially sparse observations; joint spatiotemporal inference for time-dependent PDEs is a natural extension.
6. **Foundation model potential**: Pre-training a unified function-space diffusion backbone across multiple PDE types and physical domains is a promising research direction.

## Related Work & Insights

| Method | Space | Conditioning | Resolution-Invariant | Posterior Sampling | Training Cost |
|---|---|---|---|---|---|
| **FunDPS (Ours)** | Function space | Plug-and-play guidance | ✅ | ✅ | Moderate (−75% via curriculum) |
| DiffusionPDE | Pixel space (fixed res.) | Plug-and-play guidance | ❌ | ✅ | High |
| DDO (Lim et al.) | Function space | None (unconditional only) | ✅ | ❌ | Moderate |
| Baldassari et al. | Function space | Conditional score model | ✅ | ✅ | High (retraining per observation) |
| FNO / DeepONet | Function space | End-to-end mapping | ✅ | ❌ (deterministic) | Low |
| PINN | Discrete | PDE residual constraint | ❌ | ❌ (deterministic) | Moderate |
| Kerrigan et al. (FFM) | Function space | Conditional flow matching | ✅ | ✅ | High (task-specific training) |

**Key distinction from DiffusionPDE**: DiffusionPDE models the $128 \times 128$ pixel space and requires complete retraining when the resolution changes. FunDPS models the function space via neural operators and natively supports cross-resolution inference. FunDPS also employs GRF noise (rather than multivariate Gaussian) to ensure function-space consistency, resulting in smoother guidance and reducing the required steps from 2000 to 200–500.

**Key distinction from DDO**: DDO supports only unconditional generation. FunDPS combines the infinite-dimensional Tweedie formula with Bayesian likelihood decomposition to enable plug-and-play conditional guidance at inference time, allowing a single model to handle forward, inverse, and mixed problems.

**Key distinction from deterministic solvers**: FNO, PINN, and similar methods fail completely under 3% sparse observations (errors of 30%–200%+), as they produce deterministic point estimates and cannot leverage prior regularization. FunDPS naturally handles ill-posedness through posterior sampling.

**Further Connections**

1. **The essential difference between function space and pixel space**: The core insight is that physical systems are described by continuous functions, and modeling them in discrete pixel space constitutes a level mismatch. The combination of GRF noise and neural operators ensures consistent diffusion behavior across changes in discretization—a principle transferable to generative tasks over any continuous field (weather fields, electromagnetic fields, stress fields, etc.).

2. **Value of plug-and-play in scientific computing**: Analogous to DPS/MCG in computer vision, training a prior model once enables handling of diverse observation configurations. For industrial settings with frequently changing sensor layouts (e.g., oil-and-gas exploration, environmental monitoring), this implies substantial deployment flexibility.

3. **Generality of multi-resolution curriculum learning**: The coarse-to-fine training strategy reduces GPU time by 75% and is transferable to other operator-learning-based generative models such as functional flow matching.

4. **Infinite-dimensional extension of the Tweedie formula**: This theoretical contribution is independent of the FunDPS framework and applicable to other settings requiring conditional expectation estimation in function space, such as classifier-free guidance or RLHF in infinite-dimensional settings.

5. **Convergence with foundation models**: The authors propose a function-space diffusion foundation model pre-trained across PDE types in the outlook section, which aligns closely with the vision of foundation models for science. With large-scale multi-physics pre-training, such a model could serve as a general-purpose posterior sampling engine for scientific computing.

6. **Research opportunities from limitations**: Adaptive tuning of guidance weights $\boldsymbol{\zeta}$, improved Tweedie approximations at high noise levels, and joint spatiotemporal inference are all valuable directions for future work.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Rigorously extending the Tweedie formula to infinite-dimensional Banach spaces constitutes a solid theoretical contribution; the plug-and-play posterior sampling framework in function space is the first of its kind for PDE inverse problems.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 5 PDEs × forward/inverse = 10 subtasks, covering linear/nonlinear equations and diverse boundary conditions; ablation studies are complete; however, evaluation is limited to $128 \times 128$ resolution without validation on real physical scenarios.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous with consistent notation; figures and tables are clear; the theory–method–experiment logical chain is complete.
- **Value**: ⭐⭐⭐⭐½ — Provides a theoretically grounded and practically efficient posterior sampling framework for inverse problems in scientific computing, with the potential to become a standard tool for PDE inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](split_gibbs_discrete_diffusion_posterior_sampling.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)
- [\[NeurIPS 2025\] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)
- [\[NeurIPS 2025\] Tree-Guided Diffusion Planner](tree-guided_diffusion_planner.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)

</div>

<!-- RELATED:END -->
