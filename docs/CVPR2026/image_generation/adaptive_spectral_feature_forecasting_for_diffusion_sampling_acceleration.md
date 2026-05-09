---
title: >-
  [Paper Note] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration
description: >-
  [CVPR 2026][Image Generation][Diffusion Sampling Acceleration] Ours proposes Spectrum, a global spectral domain feature forecasting method based on Chebyshev polynomials. By treating the intermediate features of the diffusion denoiser as functions of time and fitting coefficients via ridge regression, it achieves long-range feature forecasting where errors do not grow with step size. It reaches 4.79× acceleration on FLUX.1 and 4.67× on Wan2.1-14B with near-lossless quality.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Sampling Acceleration
  - Feature Caching
  - Chebyshev Polynomials
  - Spectral Methods
  - training-free
date: 2026-05-08
content_hash: 8cd80581194d6b51
---

# Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration

**Conference**: CVPR 2026  
**arXiv**: [2603.01623](https://arxiv.org/abs/2603.01623)  
**Code**: [GitHub](https://hanjq17.github.io/Spectrum)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Diffusion Sampling Acceleration, Feature Caching, Chebyshev Polynomials, Spectral Methods, training-free

## TL;DR

Ours proposes Spectrum, a global spectral domain feature forecasting method based on Chebyshev polynomials. By treating the intermediate features of the diffusion denoiser as functions of time and fitting coefficients via ridge regression, it achieves long-range feature forecasting where errors do not grow with step size. It reaches 4.79× acceleration on FLUX.1 and 4.67× on Wan2.1-14B with near-lossless quality.

## Background & Motivation

**State of the Field**: Diffusion models (especially Diffusion Transformers) generate high-quality images/videos, but inference requires dozens to hundreds of denoiser forward passes, which is computationally expensive.  
**Limitations of Prior Work**: Among existing acceleration schemes, **feature caching** avoids extra training by caching features at selected timesteps and reusing them in subsequent steps to skip expensive computations.  

However, existing caching methods rely on **local approximation**:
- **Naive reusing**: Directly copying the most recently cached features, which oversimplifies temporal dynamics.
- **TaylorSeer**: Local forecasting based on discrete Taylor expansion, but its error grows at a rate of $((j-k)\delta_t)^{P+1}$—**the larger the step size, the larger the error**, leading to severe quality degradation at high acceleration ratios.

**Root Cause**: High acceleration ratios require large skipping intervals, and the error of local predictors exacerbates precisely during large intervals. The authors identify the worst-case error of Taylor predictors through theoretical analysis and point out their fundamental limitation: the inability to capture the global long-range dynamics of the sampling trajectory.

**Starting Point**: **Shifting from local approximation in the time domain to global modeling in the frequency domain**. Each feature channel of the denoiser output is treated as a function of time and approximated globally using Chebyshev polynomials—a set of orthogonal bases with excellent numerical properties—thereby breaking the error bottleneck of local forecasting.

## Method

### Overall Architecture

The core workflow of Spectrum: In a diffusion sampling process of $N$ steps, a subset of timesteps $\mathbb{U}$ is selected to execute actual network forward passes, while for the remaining steps $\mathbb{V} = \mathbb{T} \setminus \mathbb{U}$, the spectral domain predictor is used instead. This is an **online fitting-then-forecasting** process.

### Key Designs

1.  **Chebyshev Polynomial Spectral Decomposition**:
    -   **Function**: Treats each channel of the denoiser output features $\mathbf{h}_t = [h_1(t), \cdots, h_F(t)]$ as a time function and approximates it using $M$-th order Chebyshev polynomials:
        $h_i(t) = \sum_{m=0}^{M} c_{m,i} T_m(\tau), \quad \tau = 2t - 1$
    -   **Design Motivation**: Chebyshev polynomials form an orthogonal basis whose approximation error is controlled by the polynomial order $M$ and is independent of the step size—maintaining controllable accuracy even when forecasting far into the future. According to Theorem 3.2, for functions analytically extendable to a Bernstein ellipse, the truncation error of the Chebyshev series decays exponentially as $\rho^{-M}$.

2.  **Online Ridge Regression Coefficient Fitting**:
    -   **Function**: Utilizes already cached feature points to fit Chebyshev coefficients online.
    -   **Mechanism**: Constructs a design matrix $\mathbf{\Phi}_{t_j}$ and feature matrix $\mathbf{H}_{t_j}$, solving a ridge regression problem:
        $\mathbf{C}_{t_j} = (\mathbf{\Phi}_{t_j}^\top \mathbf{\Phi}_{t_j} + \lambda \mathbf{I})^{-1} \mathbf{\Phi}_{t_j}^\top \mathbf{H}_{t_j}$
    -   The dimension of the matrix inverse is only $(M+1) \times (M+1)$, making computational overhead negligible when $M$ is small (solved via Cholesky decomposition).
    -   **Regularization term $\lambda$**: Prevents overfitting and enhances numerical stability; experiments confirm its critical role.

3.  **Adaptive Timestep Scheduling**:
    -   **Function**: Executes actual forward passes more densely in the early stages of sampling and gradually increases the predictor usage ratio later.
    -   **Mechanism**: Selects $\mathbb{U} = \{\tau_j : j = \lfloor\alpha \frac{r(r+1)}{2}\rfloor\}$, where intervals grow with $r$.
    -   **Design Motivation**: Errors in early steps propagate and amplify through ODE integration; hence, more actual network computation is needed early on to ensure foundational accuracy.

4.  **Cache Final Layer Only**:
    -   **Function**: Instantiates Spectrum only for the output of the final attention block rather than caching layer by layer.
    -   **Design Motivation**: Original TaylorSeer caches every layer, introducing $L$ times extra overhead; experiments find that caching only the final layer yields comparable or even superior quality.

### Theoretical Analysis

**Core Theorem (Theorem 3.3)**: The error bound of Spectrum does not depend on the step size $\tau_j - \tau_k$. Instead, it is controlled by the polynomial order $M$, the minimum singular value of the design matrix $\sigma_{\min}(\mathbf{\Phi})$, and the regularization strength $\lambda$. This stands in stark contrast to the Taylor method's error $\propto ((j-k)\delta_t)^{P+1}$.

## Key Experimental Results

### Main Results I: Text-to-Image Generation (DrawBench, Table 1)

| Method | FLUX Speedup | FLUX PSNR↑ | FLUX SSIM↑ | FLUX LPIPS↓ | FLUX ImageReward↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 50 steps (ref) | 1.00× | - | - | - | 1.00 |
| TaylorSeer (N=4,O=1) | 3.13× | 22.31 | 0.841 | 0.215 | 0.99 |
| TaylorSeer (N=4,O=2) | 3.03× | 20.76 | 0.812 | 0.247 | 1.02 |
| **Ours (α=0.75)** | **3.47×** | **24.32** | **0.854** | **0.217** | 0.99 |
| TaylorSeer (N=6,O=1) | 4.14× | 20.24 | 0.785 | 0.294 | 1.00 |
| **Ours (α=3.0)** | **4.79×** | **22.21** | **0.788** | **0.261** | 1.00 |

### Main Results II: Text-to-Video Generation (VBench, Table 2)

| Method | Wan2.1-14B Speedup | PSNR↑ | SSIM↑ | VBench Quality↑ |
| :--- | :--- | :--- | :--- | :--- |
| 50 steps (ref) | 1.00× | - | - | 83.15 |
| TaylorSeer (N=4,O=1) | 3.01× | 19.46 | 0.660 | 82.74 |
| **Ours (α=0.75)** | **3.40×** | **22.78** | **0.749** | **82.80** |
| TaylorSeer (N=6,O=1) | 3.94× | 17.24 | 0.585 | 81.38 |
| **Ours (α=3.0)** | **4.67×** | **21.24** | **0.694** | **82.21** |

In high acceleration scenarios (4–5×), Spectrum maintains a 2–4 dB PSNR advantage over TaylorSeer.

### Ablation Study

-   **Regularization strength $\lambda$**: Performance is poor at $\lambda = 0$; $\lambda = 0.1$ is optimal—regularization is crucial to prevent overfitting.
-   **Polynomial order $M$**: $M = 4$ is sufficient; higher orders provide no significant gain.
-   **Adaptive scheduling vs. fixed interval**: Adaptive scheduling outperforms fixed intervals by 1–2 dB PSNR at high acceleration ratios.
-   **Cache final layer only vs. layer-wise**: Caching only the final layer not only saves memory but also yields slightly better results.

### Key Findings

-   Taylor predictors exaggerate local details but lose global semantics at high acceleration ratios; Spectrum maintains color consistency and semantic correctness.
-   The computational overhead of Spectrum is negligible compared to the network forward pass (complexity dominated by $O(K(M+1)F)$, where $K$ and $M$ are small).
-   The method is effective for both image and video diffusion models and is compatible with different ODE solvers.

## Highlights & Insights

1.  **Paradigm Shift from Local to Global**: Advancing feature caching from local time-domain approximation to global spectral-domain modeling represents a methodological leap.
2.  **Theoretical Guarantees**: The theorem stating that error does not accumulate with step size is the core theoretical contribution, providing confidence for high-acceleration scenarios.
3.  **Engineering Simplicity**: Only requires ridge regression for coefficient fitting and Cholesky decomposition for inversion, with minimal extra overhead.
4.  **Broad Applicability**: Validated across four SOTA models: FLUX.1, SD3.5-Large, Wan2.1-14B, and HunyuanVideo.

## Limitations & Future Work

-   Requires at least $M+1$ cached points before forecasting can begin; the initial phase still requires full network execution.
-   Assumes the feature functions are analytic (extendable to a Bernstein ellipse); whether the smoothness assumption holds for all actual features remains to be verified.
-   The adaptive scheduling hyperparameter $\alpha$ requires tuning for different models.
-   Joint usage with orthogonal technologies like distillation or token pruning has not yet been explored.

## Related Work & Insights

-   **TaylorSeer**: The most direct competitor, using discrete Taylor expansion to forecast cached features.
-   **TeaCache**: A scheme that dynamically decides when to cache, complementary to Spectrum's scheduling strategy.
-   **FORA/ToCa**: Methods based on direct cache reuse, which performed worse than predictive schemes.
-   **Insight**: The classic status of Chebyshev polynomials in numerical analysis is cleverly introduced into deep learning inference acceleration, suggesting that other mathematical tools (e.g., Fourier bases, wavelet bases) may also be suitable for similar scenarios.

## Rating

-   Novelty: ⭐⭐⭐⭐⭐ First to introduce spectral domain methods into diffusion feature caching acceleration with solid theoretical analysis.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 SOTA models (image + video), two acceleration levels, and a full ablation suite.
-   Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivation, motivation naturally follows from Taylor error analysis, complete logical chain.
-   Value: ⭐⭐⭐⭐⭐ 4-5× acceleration with near-lossless quality, training-free, and high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](../../AAAI2026/image_generation/procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)
- [\[CVPR 2026\] TC-Padé: Trajectory-Consistent Padé Approximation for Diffusion Acceleration](tc-padé_trajectory-consistent_padé_approximation_for_diffusion_acceleration.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)

<!-- RELATED:END -->
