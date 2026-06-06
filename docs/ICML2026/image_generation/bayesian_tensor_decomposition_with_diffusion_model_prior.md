---
title: >-
  [Paper Note] Bayesian Tensor Decomposition with Diffusion Model Prior
description: >-
  [ICML2026][Image Generation][Tensor Decomposition] DiffBCP injects a pre-trained diffusion model as an implicit data prior into Bayesian CP tensor decomposition. By employing a split Gibbs sampler for tractable posterior…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Tensor Decomposition"
  - "Diffusion Model Prior"
  - "Bayesian Inference"
  - "Image Inpainting"
  - "Automatic Rank Selection"
date: 2026-05-08
content_hash: 4762394c23638031
---

# Bayesian Tensor Decomposition with Diffusion Model Prior

**Conference**: ICML2026  
**arXiv**: [2606.03212](https://arxiv.org/abs/2606.03212)  
**Code**: [GitHub](https://github.com/taozerui/DiffBCP)  
**Area**: Image Restoration  
**Keywords**: Tensor Decomposition, Diffusion Model Prior, Bayesian Inference, Image Inpainting, Automatic Rank Selection

## TL;DR

DiffBCP injects a pre-trained diffusion model as an implicit data prior into Bayesian CP tensor decomposition. By employing a split Gibbs sampler for tractable posterior inference, it significantly outperforms traditional and deep tensor decomposition baselines on image inpainting and denoising tasks (e.g., PSNR improvement up to +2.33 dB on FFHQ).

## Background & Motivation

**Background**: Low-rank tensor decomposition (TD) is a classic tool for multidimensional data analysis, achieving efficient representation and compression by decomposing high-order tensors into contractions of small factors. When data is complete and clean, even the simplest CP decomposition can yield satisfactory results.

**Limitations of Prior Work**: When observed data contains severe missingness or noise, the low-rank assumption as the sole structural prior becomes insufficient. Existing methods typically add hand-crafted priors (such as sparsity or smoothness), but these fail to capture the rich statistical characteristics of real-world data. Non-linear TD methods (e.g., DeepTensor) introduce deep network structures but lack a probabilistic modeling framework, while methods using fixed denoising networks as priors (e.g., GLON) are unstable under high missing rates.

**Key Challenge**: Diffusion models, currently the strongest data-driven priors, cannot be directly integrated with tensor decomposition and tractable posterior inference. Diffusion priors are implicitly defined (via score functions) and are coupled with the CP factor likelihood and low-rank constraints, causing standard sampling methods to fail.

**Goal**: Design a probabilistic framework to unify structural low-rank priors with learned diffusion model priors within Bayesian tensor decomposition, while achieving automatic rank selection and tractable posterior sampling.

**Key Insight**: The authors observe that noise precision $\tau$ and the coupling parameter $\rho$ always appear jointly in the likelihood term. By setting $\tau \rho^2 = c$ (a constant), $\rho$ automatically adjusts to maintain the relative scale between the likelihood and the coupling term when $\tau$ is inferred during sampling.

**Core Idea**: Use auxiliary variables to decouple the joint distribution into two independent sub-steps: "conjugate update of CP factors + diffusion-guided denoising," realizing Bayesian tensor decomposition with hybrid priors.

## Method

### Overall Architecture

The input to DiffBCP is an observed tensor $\mathscr{Y}$ with missingness and noise (e.g., a partially missing noisy image), and the output is the restored complete tensor $\mathscr{X}$. The overall framework is divided into three layers: (1) A probabilistic CP decomposition model defines the joint distribution, combining data likelihood, low-rank CP constraints, CUSP shrinkage priors, and diffusion model priors; (2) A split Gibbs sampler introduces an auxiliary variable $\mathscr{Z}$ to decouple the complex joint posterior into two alternating sampling steps; (3) A noise-adaptive coupling schedule automatically adjusts $\rho$, eliminating the need for manual annealing tuning. In each iteration, all CP factors are updated via conjugate sampling, followed by an auxiliary variable update through diffusion denoising. Posterior samples are collected after a burn-in period.

### Key Designs

1.  **Hybrid Prior Probabilistic CP Model**:

    - **Function**: Encodes both structural low-rank constraints and data-driven priors within a single probabilistic framework.
    - **Mechanism**: The joint distribution $p(\mathscr{Y}, \mathscr{X}, \boldsymbol{\lambda}, \mathbf{A}^{(1:N)}, \tau)$ involves three types of priors: (a) Structural constraints $p(\mathscr{X} | \mathbf{A}, \boldsymbol{\lambda}) = \delta(\mathscr{X} - \mathrm{CP}(\boldsymbol{\lambda}, \mathbf{A}^{(1:N)}))$ to enforce low rank; (b) CUSP shrinkage process priors $\lambda_r | \theta_r \sim \mathcal{N}(0, \theta_r)$, which shrink component weights toward zero as the rank $r$ increases to achieve automatic rank determination; (c) A pre-trained diffusion model $\nabla_{\mathscr{X}_t} \log p(\mathscr{X}_t; \sigma(t)) = s_\psi(\mathscr{X}_t, t)$ serving as an implicit prior for the reconstructed tensor. It is theoretically proven that the CUSP prior causes the tail probability of the $r$-th component to decay at a rate of $(\beta/(1+\beta))^r$.
    - **Design Motivation**: Low-rank priors capture global structure but lack textural details; diffusion priors capture rich data distributions but lack low-rank inductive biases. The two are complementary, particularly in high missing rate scenarios.

2.  **Split Gibbs Sampler**:

    - **Function**: Facilitates tractable posterior inference where an implicit diffusion prior is coupled with low-rank constraints.
    - **Mechanism**: An auxiliary variable $\mathscr{Z}$ is introduced, and a coupling term $\phi(\mathscr{Z}, \mathscr{X}; \rho) = \frac{1}{2\rho^2}\|\mathscr{Z} - \mathscr{X}\|_F^2$ is added to decouple the joint distribution into two sub-problems: (a) CP factor updates given $\mathscr{Z}$—where $\boldsymbol{\lambda}$, $\mathbf{A}^{(n)}$, and $\tau$ have closed-form conjugate distributions for direct sampling; (b) Auxiliary variable updates given $\mathscr{X}$—equivalent to a denoising problem with observation $\mathscr{X}$ and noise level $\rho$, solved using the diffusion model SDE from $\sigma(T) = \rho$ back to $t=0$. Theoretical analysis shows that as $\rho \to 0$, the smoothed posterior converges to the original posterior (TV distance vanishes), though excessively small $\rho$ makes denoising more difficult, representing a bias-variance trade-off.
    - **Design Motivation**: Directly sampling from the original joint distribution is infeasible because the gradient of the implicit prior required for Langevin samplers cannot be calculated; split Gibbs decomposes the difficult joint sampling into two solvable sub-problems.

3.  **Noise-Adaptive Coupling Schedule**:

    - **Function**: Automatically determines the coupling parameter $\rho$, eliminating manual annealing tuning.
    - **Mechanism**: By setting $\tau \rho^2 = c$ (where $c$ is a constant hyperparameter), $\rho$ is adaptively adjusted because $\tau$ is automatically inferred in each Gibbs iteration from the conjugate Gamma distribution $\tau | \cdots \sim \mathrm{Gamma}(\alpha_0 + |\Omega|/2, \kappa_0 + \frac{1}{2}\sum(y - x)^2)$. This is more robust than the deterministic annealing strategies used in methods like PnP-DM.
    - **Design Motivation**: Methods like PnP-DM are highly sensitive to the choice of $\rho_{\min}$, and incorrect scheduling can lead to significant performance degradation; in a fully Bayesian framework, $\tau$ is learned from the data, making it naturally suited to drive the scheduling of $\rho$.

## Key Experimental Results

### Main Results (FFHQ + ImageNet Image Inpainting/Denoising)

Evaluations were performed on $256 \times 256$ images. For each dataset, 128 test images were randomly selected, and Gaussian noise with $\sigma=0.05$ was added:

| Dataset / Mask | Metric | DiffBCP | DeepTensor (Strongest Baseline) | BCP | Gain |
|---------------|------|---------|----------------------|-----|------|
| FFHQ / Uniform(0.7) | PSNR↑ | **32.13** | 28.23 | 26.28 | +3.90 |
| FFHQ / Uniform(0.9) | PSNR↑ | **28.28** | 26.11 | 21.61 | +2.17 |
| FFHQ / Stripe | PSNR↑ | **27.91** | 26.44 | 9.26 | +1.47 |
| FFHQ / Irregular | PSNR↑ | **30.34** | 28.01 | 22.64 | +2.33 |
| ImageNet / Uniform(0.7) | PSNR↑ | **28.95** | 26.03 | 24.34 | +2.92 |
| ImageNet / Irregular | PSNR↑ | **27.02** | 25.16 | 21.33 | +1.86 |
| ImageNet / Avg | SSIM↑ | **78.92** | 66.50 | — | +12.42 |

DiffBCP achieved the best performance across all datasets and mask types, and LPIPS was also consistently superior (e.g., FFHQ Irregular: 15.98 vs DeepTensor 26.19). GLON proved highly unstable at high missing rates, often converging to all zeros.

### High-Resolution OOD Results

Evaluations were performed on $2048 \times 2048$ out-of-distribution images (using a diffusion prior trained on $256 \times 256$):

| Image / Mask | Metric | DiffBCP | PuTT | BCP |
|-------------|------|---------|------|-----|
| Marseille / Uniform(0.9) | PSNR↑ | **20.15** | 19.63 | 16.94 |
| Tokyo / Uniform(0.95) | PSNR↑ | **18.90** | 18.33 | 16.90 |
| Westerlund / Irregular | PSNR↑ | **25.27** | 24.38 | 22.26 |
| Tokyo / Irregular | SSIM↑ | **51.38** | 45.03 | 40.40 |

Even under significant distribution shift, DiffBCP remains superior to PuTT, as the inductive bias provided by the low-rank structure partially compensates for the distribution mismatch. PnP-DM failed completely on high-resolution images.

## Highlights & Insights

- This is the first fully probabilistic framework to introduce a pre-trained diffusion model as a data prior into Bayesian tensor decomposition, extending the plug-and-play paradigm to the field of tensor decomposition.
- The CUSP prior enables bidirectional rank adaptation: shrinking redundant components and adding new ones when necessary, making the model robust to the initial rank setting.
- Low-rank constraints make the posterior distribution easier to sample (faster mixing) while providing structural inductive bias for OOD generalization.
- Theoretical analysis provides a bias bound for the split Gibbs sampler (Theorem 3.4), revealing the bias-variance trade-off in the selection of $\rho$.

## Limitations & Future Work

- Performance is dependent on the low-rank structure assumption of the underlying signal; if the data is not low-rank, the contribution of the CP module diminishes.
- The current implementation needs to handle the full tensor, leading to high memory overhead for extremely large tensors; stochastic mini-batch updates are a direction for improvement.
- Only CP decomposition is used, and other forms like tensor train or tensor ring, which might be better suited for specific data patterns, have not been explored.
- Validated only on image inpainting and denoising; not yet extended to other inverse problems such as compressed sensing or super-resolution.

## Related Work & Insights

- **PnP-DM (Wu et al., 2024)**: Also uses a split Gibbs sampler with diffusion priors but lacks structural tensor decomposition constraints, making it unstable for high-resolution and high-missing-rate scenarios.
- **DPS (Chung et al., 2023)**: Diffusion posterior sampling, but employs approximate gradient guidance rather than exact Bayesian inference.
- **GLON (Zhao et al., 2022)**: TD combined with pre-trained denoising networks, but the denoiser is far less powerful than diffusion models and the framework is non-probabilistic.
- **DeepTensor (Saragadam et al., 2024)**: TD with deep network structures, generating finer details but suffering from artifacts.
- Insight: The approach of injecting powerful generative priors into traditional structured models can be extended to other structured signal recovery problems.

## Rating
- Novelty: 8/10 — Successfully integrates diffusion model priors into Bayesian tensor decomposition for the first time, with innovations in both theory and algorithm design.
- Experimental Thoroughness: 8/10 — Covers multiple datasets, mask types, OOD, and high-resolution scenarios, including theoretical analysis and ablations.
- Writing Quality: 8/10 — Mathematical derivations are clear, and theory is closely integrated with experiments.
- Value: 7/10 — Opens a new direction for tensor decomposition + generative models, although the practical application scenarios are relatively specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery](../../CVPR2026/image_generation/reparameterized_tensor_ring_functional_decomposition_for_multi-dimensional_data_.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](../../ICCV2025/image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[AAAI 2026\] Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](../../AAAI2026/image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)
- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](../../ICCV2025/image_generation/learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)

</div>

<!-- RELATED:END -->
