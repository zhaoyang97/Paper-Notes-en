---
title: >-
  [Paper Note] Linearly Constrained Diffusion Implicit Models
description: >-
  [NeurIPS 2025][3D Vision][diffusion models] This paper proposes CDIM, a DDIM-based algorithm for solving linear inverse problems. By aligning the residual energy with the $\chi^2$ distribution of the forward diffusion process, CDIM adaptively controls the number and step size of projection steps, achieving inference speeds 10–50× faster than DPS while exactly satisfying measurement constraints in the noiseless case.
tags:
  - NeurIPS 2025
  - 3D Vision
  - diffusion models
  - inverse problems
  - linear constraints
  - DDIM
  - accelerated sampling
date: 2026-05-08
content_hash: 5a89dab5f08430c1
---

# Linearly Constrained Diffusion Implicit Models

**Conference**: NeurIPS 2025
**arXiv**: [2411.00359](https://arxiv.org/abs/2411.00359)
**Code**: [https://grail.cs.washington.edu/projects/cdim/](https://grail.cs.washington.edu/projects/cdim/)
**Area**: 3D Vision / Image Restoration
**Keywords**: diffusion models, inverse problems, linear constraints, DDIM, accelerated sampling

## TL;DR

This paper proposes CDIM, a DDIM-based algorithm for solving linear inverse problems. By aligning the residual energy with the $\chi^2$ distribution of the forward diffusion process, CDIM adaptively controls the number and step size of projection steps, achieving inference speeds 10–50× faster than DPS while exactly satisfying measurement constraints in the noiseless case.

## Background & Motivation

Recovering a signal $\mathbf{x}$ from noisy linear measurements $\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\sigma}$ is a fundamental problem encompassing tasks such as super-resolution, inpainting, and denoising. Using pretrained diffusion models as priors to solve such inverse problems has become a prominent research direction.

**Limitations of Prior Work**:

**Excessive projection steps**: Methods such as DPS alternate between unconditional denoising and constraint projection at each diffusion step, but each projection requires a network forward pass, resulting in extremely slow inference (>70 seconds).

**Constraint failure under accelerated sampling**: Directly combining DPS with DDIM accelerated sampling leads to insufficient projection due to large step sizes, producing blurry or divergent outputs.

**Inability to exactly satisfy constraints**: DPS employs fixed small projection step sizes, which cannot guarantee $\mathbf{A}\hat{\mathbf{x}}_0 = \mathbf{y}$ even in the noiseless setting.

**Difficulty in step size selection**: Projection step sizes that are too large cause divergence or out-of-distribution drift, while those that are too small result in insufficient convergence.

**Key Challenge**: How can the number of projection steps be substantially reduced without either over-projecting (pulling samples out of distribution) or under-projecting (failing to satisfy constraints)?

**Core Idea**: The analytic $\chi^2$ distribution of the residual energy $\|\mathbf{A}\mathbf{x}_t - \mathbf{y}\|^2$ under the forward diffusion process is leveraged to guide the projection strategy — projection is performed only when the residual exceeds a "plausible region," and the step size is determined via one-dimensional search so that the residual lands precisely on the boundary of this region.

## Method

### Overall Architecture

The CDIM inference procedure (Algorithm 1):
1. Initialize $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. For each timestep $t = T, T-\delta, \ldots, 1$:
    - Perform one unconditional DDIM denoising step to obtain $\mathbf{x}_{t-\delta}$
    - Compute the plausible region boundary $\rho_t(\mathbf{y}) = \mu_{t-\delta}(\mathbf{y}) + c \cdot \sigma_{t-\delta}(\mathbf{y})$
    - While $\|\mathbf{A}\mathbf{x}_{t-\delta} - \mathbf{y}\|^2 > \rho_t$: compute the Tweedie estimate $\hat{\mathbf{x}}_0$, compute gradient $\mathbf{g}$, find optimal step size $\eta^*$ via one-dimensional search, and update $\mathbf{x}_{t-\delta}$

### Key Designs

1. **Surrogate Residual and $\chi^2$ Distribution**:

    - **Function**: Provides an analytically computable stopping criterion for projection.
    - **Mechanism**: The distribution of the true objective $L_t = \|\mathbf{A}\hat{\mathbf{x}}_0(\mathbf{x}_t) - \mathbf{y}\|^2$ is intractable (depending on the data distribution). However, the surrogate residual $R_t = \|\mathbf{A}\mathbf{x}_t - \mathbf{y}\|^2$ follows a non-central generalized $\chi^2$ distribution under the forward process, with analytically computable mean and variance:
        - $\mu_t(\mathbf{y}) = (\sqrt{\bar\alpha_t} - 1)^2 \|\mathbf{y}\|^2 + (1 - \bar\alpha_t) \text{tr}(\mathbf{A}\mathbf{A}^\top)$
        - $\sigma_t^2(\mathbf{y})$ is similarly computable in closed form.
    - **Proposition 1**: When $|R_t - \mathbb{E}[R_t|\mathbf{y}]| \leq \gamma$, it follows that $|L_t - \mathbb{E}[L_t|\mathbf{y}]| \leq \gamma/\bar\alpha_t + O(\sqrt{1-\bar\alpha_t}/\bar\alpha_t)$. That is, controlling the surrogate residual implies control over the true objective.
    - **Design Motivation**: Computing $R_t$ requires only matrix multiplication without network evaluation, enabling step size search at zero additional NFE (Neural Function Evaluations).

2. **Adaptive Step Size Selection**:

    - **Function**: Identifies the optimal projection step size without additional NFEs.
    - **Mechanism**: The gradient $\mathbf{g} = \nabla_{\mathbf{x}_{t-\delta}} \|\mathbf{A}\hat{\mathbf{x}}_0 - \mathbf{y}\|^2$ is precomputed, and a one-dimensional search finds $\eta^* = \arg\min_\eta |\rho_{t-\delta}(\mathbf{y}) - \|\mathbf{A}(\mathbf{x}_{t-\delta} - \eta\mathbf{g}) - \mathbf{y}\|^2|$.
    - The objective is quadratic in $\eta$, making the search efficient and stable.
    - **Design Motivation**: Pulling the residual to the boundary of the plausible region (rather than its center) yields better empirical results.

3. **Stopping Criterion and Hyperparameter $c$**:

    - **Function**: Determines when to stop projection.
    - **Mechanism**: Projection halts when $\|\mathbf{A}\mathbf{x}_{t-\delta} - \mathbf{y}\|^2 \leq \rho_{t-\delta}(\mathbf{y})$.
    - Smaller $c$ (narrower plausible region) yields better results but requires more projection steps. Experiments use $c = 0.1$.
    - At high noise levels the plausible region is large and almost no projection is needed; at low noise levels the region narrows and more projection steps are required.

4. **Poisson Noise Handling**:

    - **Function**: Extends the framework to non-Gaussian noise.
    - **Mechanism**: Pearson residuals $R(\mathbf{A}\hat{\mathbf{x}}_0, \mathbf{y}) = \frac{\lambda(\mathbf{y} - \mathbf{A}\hat{\mathbf{x}}_0)}{\sqrt{\lambda\hat{\mathbf{x}}_0}}$ transform Poisson noise into approximately standard normal noise, after which the same framework is applied.

### Loss & Training

- No training is required; a pretrained diffusion model is used directly.
- Only two hyperparameters: number of denoising steps $T'$ and stopping criterion constant $c$.
- Defaults: $T' = 50$, $c = 0.1$.
- As $t \to 0$, $\hat{\mathbf{x}}_0 \to \mathbf{x}_t$ and $L_t$ becomes a simple convex quadratic, enabling exact convergence to zero.

## Key Experimental Results

### Main Results

**FFHQ 256×256 ($\sigma=0.05$)**:

| Method | Super-res FID | Inpainting (box) FID | Gaussian Deblur FID | Inpainting (random) FID | Time (s) |
|--------|--------------|---------------------|--------------------|-----------------------|---------|
| DPS | 39.35 | 33.12 | 44.05 | 21.19 | 70.42 |
| FPS-SMC | 26.62 | 26.51 | 29.97 | 33.10 | 116.90 |
| DDRM | 62.15 | 42.93 | 74.92 | 69.71 | 2.0 |
| **CDIM (T'=25)** | 33.87 | 27.51 | 34.18 | 29.67 | **2.4** |
| **CDIM (T'=50)** | **31.54** | **26.09** | **29.68** | 28.52 | 6.4 |

CDIM achieves comparable or superior quality while running ~28× faster than DPS (2.4s vs. 70.4s).

### Ablation Study

| Denoising Steps T' | Total Projection Steps | Quality | Notes |
|-------------------|----------------------|---------|-------|
| T'=25 | 31 | Good | Default fast setting |
| T'=10 | 15 | Reasonable | Still usable |
| T'=4 | 10 | Acceptable | <1s inference, 14 NFEs |

| Stopping Criterion $c$ | Quality | Projection Steps |
|------------------------|---------|-----------------|
| c=4 | Poor constraint satisfaction | Fewest |
| c=1 | Good | Moderate |
| c=0.1 | Best | More but manageable |

**DPS + DDIM Comparison**:
- DPS + DDIM (small step size): MAE 4%, constraint unsatisfied, blurry hair
- DPS + DDIM (large step size): MAE 0.3%, constraint improved but results diverge
- CDIM: MAE 0.05%, constraints exactly satisfied with natural-looking outputs

### Key Findings

1. **Exact constraint satisfaction**: In the noiseless setting, CDIM exactly satisfies $\mathbf{A}\mathbf{x}_0 = \mathbf{y}$, even for out-of-distribution observations.
2. **Acceleration is not solely from DDIM**: DPMC employs DDIM-style sampling yet still requires 200 additional NFEs, demonstrating that DDIM alone is insufficient.
3. **Fewer projections at high noise, more at low noise**: This is intuitive — early-stage Tweedie estimates are inaccurate and should not be over-projected.
4. **Extreme acceleration remains viable**: As few as T'=4 steps (<1 second) still produces reasonable results.

## Highlights & Insights

1. **Elegant mathematical framework**: The analytic distribution of the forward process is used to guide projection during reverse sampling, providing clear theoretical motivation.
2. **Clever surrogate residual design**: $R_t$ can be computed without network evaluation, making step size search cost-free in terms of NFEs.
3. **Exact constraint satisfaction**: CDIM is the first diffusion-based inverse problem solver to achieve exact constraint satisfaction (which DPS cannot).
4. **Poisson noise extension**: Pearson residuals provide a unified treatment of non-Gaussian noise within the same framework.

## Limitations & Future Work

- **Restricted to linear constraints**: For nonlinear constraints, $\mathbb{E}[h(\mathbf{x}_0)] \neq h(\mathbb{E}[\mathbf{x}_0])$, precluding extension of the Tweedie estimator.
- Requires an explicit form of $\mathbf{A}$ to compute $\|\mathbf{A}\mathbf{x}_t - \mathbf{y}\|^2$.
- For Poisson noise, the normal approximation of Pearson residuals breaks down under extreme noise levels or when $\hat{\mathbf{x}}_0$ approaches zero.
- Computing the gradient $\mathbf{g}$ still requires a network evaluation; the method substantially reduces but does not eliminate such calls.

## Related Work & Insights

- **DPS (2023)**: The primary baseline for CDIM; DPS's small per-step projection strategy fails under accelerated sampling.
- **DDRM/DDNM**: SVD-based methods are fast but achieve limited quality.
- **DSG**: Employs similar gradient-based updates but uses soft rather than exact constraints.
- **Insight**: The statistical properties of the forward process constitute a valuable source of guidance for the reverse process; replacing heuristic strategies with analytic distributions is a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Using the $\chi^2$ distribution to guide projection is an elegant and original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task, multi-dataset comparisons with ablations and novel applications (temporal rephotography / point clouds), though quantitative evaluation could be further enriched.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and clear; figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ Opens a new Pareto frontier in the speed–quality trade-off for diffusion-based inverse problem solving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](../../ICCV2025/3d_vision/spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](../../ICCV2025/3d_vision/learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)

</div>

<!-- RELATED:END -->
