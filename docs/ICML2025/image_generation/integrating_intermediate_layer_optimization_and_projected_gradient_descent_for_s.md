---
title: >-
  [Paper Note] Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models
description: >-
  [ICML2025][Image Generation][Inverse Problems] The authors propose two methods, DMILO and DMILO-PGD, which leverage intermediate layer optimization (ILO) to partition the diffusion model sampling process, thereby significantly reducing GPU memory consumption. By integrating projected gradient descent (PGD) to prevent sub-optimal convergence, these methods comprehensively outperform state-of-the-art (SOTA) methods such as DMPlug on both linear and non-linear inverse problems.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Inverse Problems"
  - "Diffusion Models"
  - "Intermediate Layer Optimization"
  - "Projected Gradient Descent"
  - "Image Reconstruction"
  - "ILO"
  - "PGD"
date: 2026-05-08
content_hash: 89228f0375c899ab
---

# Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models

**Conference**: ICML2025  
**arXiv**: [2505.20789](https://arxiv.org/abs/2505.20789)  
**Code**: [StarNextDay/DMILO](https://github.com/StarNextDay/DMILO)  
**Area**: Inverse Problem Solving / Diffusion Models  
**Keywords**: Inverse Problems, Diffusion Models, Intermediate Layer Optimization, Projected Gradient Descent, Image Reconstruction, ILO, PGD

## TL;DR

The authors propose two methods, DMILO and DMILO-PGD, which leverage intermediate layer optimization (ILO) to partition the diffusion model sampling process, thereby significantly reducing GPU memory consumption. By integrating projected gradient descent (PGD) to prevent sub-optimal convergence, these methods comprehensively outperform state-of-the-art (SOTA) methods such as DMPlug on both linear and non-linear inverse problems.

## Background & Motivation

The goal of inverse problems is to recover a signal $\boldsymbol{x}^*$ from noisy observations $\boldsymbol{y} = \mathcal{A}(\boldsymbol{x}^*) + \boldsymbol{\epsilon}$, which is widely applicable to fields such as medical imaging, compressive sensing, and remote sensing. Diffusion models (DMs) have emerged as powerful generative priors and have achieved SOTA performance in solving inverse problems. However, existing methods suffer from two major limitations:

**Limitations of Prior Work**:
- **GPU Memory Bottleneck**: CSGM-type methods (like DMPlug) require retaining the entire computational graph during the sampling process for backpropagation. Consequently, the memory consumption scales linearly with the number of sampling steps, resulting in Out-Of-Memory (OOM) issues on an RTX 4090 even with just 4 steps.
- **Sub-optimal Convergence**: These methods rely heavily on the selection of the initial vector, making them susceptible to trapping in local optima.

Although previous ILO approaches succeeded on GANs, their layer decomposition relied heavily on specific network architectures, rendering them difficult to generalize. This paper discovers that the diffusion sampling process naturally consists of multiple function compositions, which is highly suited for ILO-style decomposition.

## Method

### Core Idea: Treating Diffusion Sampling as Function Composition

The generation process of a diffusion model can be formulated as a composition of functions over $N$ sampling steps:

$$\mathcal{G}(\cdot) = g_1 \circ g_2 \circ \cdots \circ g_N(\cdot)$$

Each $g_i$ corresponds to a single step of DDIM sampling:

$$g_i(\boldsymbol{x}) = \frac{\sigma_{t_{i-1}}}{\sigma_{t_i}} \boldsymbol{x} + \sigma_{t_{i-1}} \left( \frac{\alpha_{t_{i-1}}}{\sigma_{t_{i-1}}} - \frac{\alpha_{t_i}}{\sigma_{t_i}} \right) \boldsymbol{x}_\theta(\boldsymbol{x}, t_i)$$

This decomposition is independent of the denoising network architecture and can be seamlessly integrated with any DM.

### DMILO: Intermediate Layer Optimization + Sparse Deviation

For the first layer (which is directly associated with the observation), the optimization problem is:

$$\hat{\boldsymbol{x}}_{t_1}, \hat{\boldsymbol{\nu}}_{t_1} = \arg\min_{\boldsymbol{x}, \boldsymbol{\nu}} \| \boldsymbol{y} - \mathcal{A}(g_1(\boldsymbol{x}) + \boldsymbol{\nu}) \|_2^2 + \lambda \|\boldsymbol{\nu}\|_1$$

For each subsequent layer, the optimization targets the results from the previous layer:

$$\hat{\boldsymbol{x}}_{t_i}, \hat{\boldsymbol{\nu}}_{t_i} = \arg\min_{\boldsymbol{x}, \boldsymbol{\nu}} \| \hat{\boldsymbol{x}}_{t_{i-1}} - (g_i(\boldsymbol{x}) + \boldsymbol{\nu}) \|_2^2 + \lambda \|\boldsymbol{\nu}\|_1$$

where $\boldsymbol{\nu}$ represents a sparse deviation term ($\ell_1$ regularization) used to explore signals outside the range of the diffusion model. Since only single-step gradient information needs to be stored at any time, the GPU memory consumption remains constant.

### DMILO-PGD: Introducing Projected Gradient Descent

Building upon DMILO, the algorithm alternately performs:

1. **Gradient Descent Step**: $\boldsymbol{x}_{t_0}^{(e)} = \boldsymbol{x}_{t_0}^{(e-1)} - \eta \nabla \|\boldsymbol{y} - \mathcal{A}(\boldsymbol{x}_{t_0}^{(e-1)})\|_2^2$
2. **Projection Step**: Projecting the updated signal back into the expanded range of the diffusion model using DMILO.

The key difference is that the projection minimizes $\|\mathcal{A}(\mathcal{G}(\boldsymbol{x}_{t_N})) - \mathcal{A}(\hat{\boldsymbol{x}}_{t_0})\|_2^2$, utilizing the forward operator $\mathcal{A}$ to guide the projection direction rather than relying purely on a distance projection as in traditional PGD. This theoretically guarantees better reconstruction.

### Theoretical Guarantees

Under Lipschitz continuity and low-dimensional manifold assumptions, Theorem 4.4 guarantees that when the number of measurements satisfies $m = \Omega(k_2 \log \frac{L_1 n}{\delta} + k^2 \log(3n))$, the measurement-optimal solution utilizing the forward operator is close to the true optimal solution:

$$\|g_1(\hat{\boldsymbol{x}}_1) - \boldsymbol{x}^*\|_2 \leq \left(1 + \frac{3}{\gamma}\right) \|g_1(\bar{\boldsymbol{x}}_1) - \boldsymbol{x}^*\|_2 + \frac{\delta}{\gamma}$$

## Key Experimental Results

The experiments cover CelebA, FFHQ, LSUN-bedroom, and ImageNet, and include 4 linear tasks and 2 non-linear tasks.

### GPU Memory Comparison (RTX 4090, Model size: 2.75GB)

| Sampling Steps | DMPlug | DMILO | DMILO-PGD |
|---------|--------|-------|-----------|
| 1 | 10.53 GB | 10.53 GB | 10.53 GB |
| 2 | 15.72 GB | 10.53 GB | 10.54 GB |
| 3 | 20.83 GB | 10.53 GB | 10.54 GB |
| 4 | N/A (OOM) | 10.54 GB | 10.54 GB |

### Super-Resolution & Inpainting (CelebA, σ=0.01)

| Method | SR PSNR↑ | SR SSIM↑ | Inpaint PSNR↑ | Inpaint SSIM↑ |
|------|---------|---------|--------------|--------------|
| DMPlug | 32.38 | 0.875 | 35.51 | 0.935 |
| DCPS | 29.47 | 0.834 | 35.42 | 0.940 |
| DMILO-PGD | **33.58** | **0.906** | **36.42** | **0.952** |

### Motion Deblurring (CelebA, σ=0.01)

| Method | FID↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|------|--------|-------|-------|
| DMPlug | 78.57 | 0.164 | 30.25 | 0.824 |
| DCPS | 35.19 | 0.054 | 31.05 | 0.856 |
| DMILO | **31.08** | **0.044** | **34.15** | **0.908** |

### Non-linear Deblurring (FFHQ, σ=0.01)

| Method | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|--------|-------|-------|
| DMPlug | 0.099 | 31.37 | 0.866 |
| DMILO-PGD | **0.047** | **34.02** | **0.919** |

## Highlights & Insights

1. **Natural Decomposition**: The function composition structure of the diffusion sampling process naturally fits the formulation of ILO without relying on specific architectures, which is a highly elegant observation.
2. **Constant GPU Memory**: Regardless of how much the number of sampling steps increases, the memory usage of DMILO remains nearly constant (~10.5 GB), whereas DMPlug scales linearly until OOM.
3. **Range Expansion via Sparse Deviation**: The $\ell_1$ regularization allows the algorithm to explore signals outside the Generative Range of the DM, which is crucial for scenarios where ground-truth signals do not fall completely within the generative distribution.
4. **Forward-Operator-Guided Projection**: Utilizing $\mathcal{A}$ to guide projections in DMILO-PGD achieves superior results compared to pure distance projection, backed by intuitive theoretical support.
5. **Significant Lead in Non-linear Tasks**: A 2.5–3.5 dB improvement in PSNR highlights the distinct advantage of the proposed method under complex degradations.

## Limitations & Future Work

1. **Limited Effectiveness of PGD on Blind Deblurring**: DMILO-PGD underperforms compared to DMILO on BID tasks. The authors hypothesize that naive gradient updates are unsuitable for kernel estimation, requiring the design of specialized kernel updating strategies.
2. **Insufficient Discussion on Computational Efficiency**: Though GPU memory usage is reduced, the computational cost might be remarkably high due to multiple rounds of outer loops × inner loops (e.g., 400 inner iterations × 10 outer iterations for super-resolution).
3. **Moderate Performance of Gaussian Deblurring on ImageNet**: The FID and LPIPS metrics underperform DCPS under certain configurations, indicating that generalization needs to be strengthened.
4. **Numerous Hyperparameters**: Hyperparameters such as $\lambda$, inner/outer learning rates, and inner/outer iteration numbers must be tuned individually for different tasks.
5. **Only DDIM Sampling Evaluated**: Integration with other samplers (such as DPM-Solver) has not yet been explored.

## Related Work & Insights

- **DMPlug** (Wang et al., 2024): The baseline that this work directly improves upon, which operates as a CSGM method optimizing the initial latent variables.
- **ILO** (Daras et al., 2021): Intermediate Layer Optimization was originally designed for GANs, and this work naturally extends it to DMs.
- **Sparse Deviation** (Dhar et al., 2018): Enables the exploration of out-of-range signals.
- **PGD for IP** (Shah & Hegde, 2018): Projected gradient descent framework.
- **DCPS** (Janati et al., 2024): Another strong competitor that performs comparably to the proposed method on certain tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of ILO and DM is natural and highly effective, and the forward-operator-guided projection offers theoretical novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely comprehensive with 6 tasks, 4 datasets, and over 9 baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured with a smooth transition between theory and experiments.
- Value: ⭐⭐⭐⭐ — Resolving the GPU memory constraint is crucial for practical deployments, although computational efficiency still has room for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[AAAI 2026\] Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes](../../AAAI2026/image_generation/constrained_particle_seeking_solving_diffusion_inverse_problems_with_just_forwar.md)
- [\[ICML 2025\] Learning Single Index Models with Diffusion Priors](learning_single_index_models_with_diffusion_priors.md)
- [\[CVPR 2025\] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](../../CVPR2025/image_generation/improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)

</div>

<!-- RELATED:END -->
