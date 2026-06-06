---
title: >-
  [Paper Note] Recurrent Memory for Online Interdomain Gaussian Processes
description: >-
  [NeurIPS 2025][Image Generation][Gaussian Processes] This paper proposes OHSVGP (Online HiPPO Sparse Variational Gaussian Process), which introduces the HiPPO (High-order Polynomial Projection Operator) framework from de…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Gaussian Processes"
  - "HiPPO"
  - "Online Learning"
  - "Long-term Memory"
  - "State Space Models"
date: 2026-05-08
content_hash: cd2444986da0425b
---

# Recurrent Memory for Online Interdomain Gaussian Processes

**Conference**: NeurIPS 2025
**arXiv**: [2502.08736](https://arxiv.org/abs/2502.08736)  
**Code**: [GitHub](https://github.com/harrisonzhu508/HIPPOSVGP)  
**Area**: Gaussian Processes / Online Learning
**Keywords**: Gaussian Processes, HiPPO, Online Learning, Long-term Memory, State Space Models

## TL;DR

This paper proposes OHSVGP (Online HiPPO Sparse Variational Gaussian Process), which introduces the HiPPO (High-order Polynomial Projection Operator) framework from deep learning into sparse variational Gaussian processes as interdomain inducing variables. By leveraging time-varying orthogonal polynomial basis functions, the method achieves long-term memory retention in online learning, with kernel matrices updated efficiently via ODE recursion.

## Background & Motivation

**Background**: Gaussian Processes (GPs) are a classical choice for time series modeling due to their expressive functional priors and principled uncertainty quantification. However, GPs face computational bottlenecks of $O(n^3)$ time and $O(n^2)$ space complexity. Sparse Variational Gaussian Processes (SVGPs) mitigate this via inducing points, and Online SVGP (OSVGP) further extends this to the online learning setting.

**Limitations of Prior Work**: OSVGP suffers from **catastrophic forgetting** in online learning. As new data arrive, inducing points inevitably drift toward the most recent data regions, causing the model to lose memory of earlier tasks. Maintaining long-term memory requires continuously increasing the number of inducing points.

**Key Insight**: The HiPPO framework is well known in deep learning for its superior long-range memory capabilities (serving as the foundation for S4 and Mamba). This paper reinterprets HiPPO's time-varying orthogonal polynomial projections as inducing variables for interdomain GPs, thereby achieving effective long-term memory retention with a fixed number of inducing variables.

## Method

### Overall Architecture

The core mechanism of OHSVGP:
1. Interpret HiPPO polynomial projection coefficients as inducing variables of an interdomain sparse variational GP.
2. Leverage HiPPO's ODE recursion to update kernel matrices incrementally, avoiding recomputation.
3. Combine with an online variational inference framework to efficiently update the posterior as new data arrive.

### Key Designs

1. **HiPPO as Interdomain Inducing Variables (Section 3.1)**: In standard interdomain GPs, inducing variables are defined as $u_m = \int f(x) \phi_m(x) dx$, where $\phi_m$ are basis functions. OHSVGP employs HiPPO's time-varying basis functions $\phi_m^{(t)}(x) = g_m^{(t)}(x) \omega^{(t)}(x)$, where $g_m^{(t)}$ are time-varying orthogonal polynomials (e.g., Legendre polynomials) and $\omega^{(t)}$ is a time-varying measure function.

    - **Design Motivation**: Conventional interdomain GPs use basis functions with fixed measures (e.g., uniform measure over a fixed interval), so new temporal indices may fall outside the predefined range. HiPPO's adaptive basis functions expand with time, naturally covering newly arrived data regions.
    - The inducing variables $u_m^{(t)} = \int f(x) \phi_m^{(t)}(x) dx$ are no longer fixed random variables but stochastic processes that evolve over time.

2. **ODE Recursive Kernel Matrix Updates (Section 3.2)**: Kernel matrices are updated efficiently using HiPPO's ODE parameters:

    - Prior cross-covariance: $\frac{d}{dt}[\mathbf{K}_{\mathbf{fu}}^{(t)}]_{n,:} = \mathbf{A}(t)[\mathbf{K}_{\mathbf{fu}}^{(t)}]_{n,:} + \mathbf{B}(t)k(x_n, t)$
    - The inducing variable covariance $\mathbf{K}_{\mathbf{uu}}^{(t)}$ involves a double integral, which is decomposed into a sum of products of two single integrals via Bochner's theorem and Random Fourier Features (RFF); each single integral can be updated via the HiPPO ODE recursion. Approximation uses 1000 RFF samples.
    - **Design Motivation**: This avoids recomputing kernel matrices upon each new data arrival, reducing the computational cost from $O(NM^2)$ to incremental updates.

3. **Extension to Multivariate Inputs (Section 3.3)**: For non-time-series data (e.g., continual learning on UCI datasets), training samples must be ordered to create a pseudo-temporal sequence. Two ordering strategies are proposed:

    - OHSVGP-o: Oracle ordering consistent with task partitioning.
    - OHSVGP-k: Heuristic ordering based on kernel similarity, $\mathbf{x}_i^{(j)} = \arg\max k(\mathbf{x}, \mathbf{x}_{i-1}^{(j)})$.

### Loss & Training

- Online ELBO (Eq. 3) is used for variational updates, with the posterior of the previous task serving as the prior for the next.
- Kernel hyperparameters are trained only on the initial task and then fixed, avoiding instability during online updates.
- For conjugate Gaussian likelihoods, the posterior admits a closed-form solution (OHSGPR), requiring no training iterations at all.

## Key Experimental Results

### Main Results 1: Time Series Prediction (NLPD↓)

| Dataset | Method | M=50 (after task 10) | M=150 (after task 10) |
|---------|--------|----------------------|----------------------|
| Solar | OSGPR | ~2.5 (catastrophic forgetting) | ~1.8 |
| Solar | OVC | ~1.2 | ~0.9 |
| Solar | OVFF | ~1.0 | ~0.8 |
| Solar | **OHSGPR** | **~0.8** | **~0.7** |
| Audio | OSGPR | severe forgetting | moderate forgetting |
| Audio | **OHSGPR** | **best** | **best** |

OSGPR begins exhibiting catastrophic forgetting around task 5, while OHSGPR maintains consistent performance throughout the learning process.

### Main Results 2: Runtime Comparison (seconds)

| Method | Solar M=50 | Solar M=150 | Audio M=100 | Audio M=200 |
|--------|-----------|-------------|-------------|-------------|
| OSGPR | 140 | 149 | 144 | 199 |
| OVC | 0.450 | 0.620 | 0.558 | 0.863 |
| OVFF | 0.327 | 0.354 | 0.295 | 0.356 |
| **OHSGPR** | **0.297** | **0.394** | **0.392** | **0.655** |

OHSGPR is **300–450× faster** than OSGPR, as it does not require optimizing inducing point locations.

### Ablation Study: Effect of Ordering Strategies in Continual Learning

| Dataset (Ordering) | OSVGP | OVC | OHSVGP-k | OHSVGP-o |
|--------------------|-------|-----|----------|----------|
| Skillcraft (1st dim) | worst | moderate | similar to OSVGP | **best** |
| Skillcraft (L2) | worst | moderate | similar to OSVGP | **best** |
| Powerplant (1st dim) | forgetting | forgetting | moderate | **best** |
| Powerplant (L2) | forgetting | moderate | moderate | **best** |

### Key Findings

- OHSVGP-o (with oracle ordering) achieves the best performance in all scenarios, highlighting that ordering strategy is critical for multivariate inputs.
- OVFF severely underfits in early tasks because its inducing variables integrate over the entire predefined time interval, diluting early information.
- Even with non-conjugate likelihoods (e.g., negative binomial for COVID data), OHSVGP outperforms baselines.
- OVC-optZ (with further optimized inducing points) exhibits performance degradation in later tasks, indicating that the online ELBO objective cannot guarantee optimal online updates of inducing point locations.

## Highlights & Insights

- Transferring HiPPO from the RNN/SSM domain into the GP framework is a natural and elegant cross-domain adaptation.
- The ODE recursive kernel matrix updates eliminate recomputation, making OHSGPR training-free in the conjugate case.
- The finite-basis reconstruction $f = \sum_{m=1}^M u_m^{(t)} g_m^{(t)}(x)$ provides interpretable function approximation as a byproduct.
- The approach is compatible with other approximate inference frameworks in SVGP (EP, Laplace), modifying only the kernel matrix computation.

## Limitations & Future Work

- RFF approximation of $\mathbf{K}_{\mathbf{uu}}^{(t)}$ may accumulate errors over long time series.
- Multivariate inputs require ordering, and the ordering strategy substantially impacts performance; OHSVGP-k does not consistently outperform OSVGP in certain scenarios.
- Kernel hyperparameters are fixed after the initial task, preventing adaptation to distributional shifts.
- The method has not been thoroughly validated on very large-scale time series or high-dimensional outputs (e.g., video generation).

## Related Work & Insights

- Unlike Markovian GPs (which also admit recursive structures), OHSVGP is specifically designed for online learning.
- VFF (Variational Fourier Features) requires a predefined time interval and is therefore unsuitable for online scenarios.
- The proposed method can potentially be combined with more advanced SSM variants (e.g., the selective mechanism of Mamba).
- Continual learning experiments on GP-VAE (ERA5 climate data) demonstrate the method's potential in deep generative models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The cross-domain transfer from HiPPO to GPs is novel, though the contribution is essentially a combination of existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three categories of tasks — time series, continual learning, and GP-VAE — with comprehensive baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and figures are intuitive, though the presentation assumes substantial background knowledge in interdomain GPs.
- **Value**: ⭐⭐⭐⭐ Addresses the core limitation of online GPs (catastrophic forgetting) with practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[CVPR 2026\] Gaussian Shannon: High-Precision Diffusion Model Watermarking Based on Communication](../../CVPR2026/image_generation/gaussian_shannon_high-precision_diffusion_model_watermarking_based_on_communicat.md)
- [\[ICCV 2025\] LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling](../../ICCV2025/image_generation/ld-rps_zero-shot_unified_image_restoration_via_latent_diffusion_recurrent_poster.md)
- [\[AAAI 2026\] ORVIT: Near-Optimal Online Distributionally Robust Reinforcement Learning](../../AAAI2026/image_generation/orvit_near-optimal_online_distributionally_robust_reinforcement_learning.md)
- [\[ICML 2026\] Adapting Noise to Data: Generative Flows from Learned 1D Processes](../../ICML2026/image_generation/adapting_noise_to_data_generative_flows_from_1d_processes.md)

</div>

<!-- RELATED:END -->
