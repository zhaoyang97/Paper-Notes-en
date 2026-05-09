---
title: >-
  [Paper Note] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][latent diffusion model] This paper proposes DWGF (Diffusion-regularized Wasserstein Gradient Flow), which rigorously formalizes posterior sampling with latent diffusion models as a regularized gradient flow of KL divergence in the Wasserstein-2 space. An ODE system in the latent space is derived to solve image inverse problems, achieving substantially higher PSNR than baselines on inpainting and super-resolution tasks on FFHQ-512.
tags:
  - NeurIPS 2025
  - Image Generation
  - latent diffusion model
  - Wasserstein gradient flow
  - inverse problem
  - posterior sampling
  - KL divergence
date: 2026-05-08
content_hash: 5bb72e87a79a10e7
---

# A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.19276](https://arxiv.org/abs/2509.19276)
**Code**: None
**Area**: Diffusion Models / Inverse Problems / Bayesian Inference
**Keywords**: latent diffusion model, Wasserstein gradient flow, inverse problem, posterior sampling, KL divergence

## TL;DR
This paper proposes DWGF (Diffusion-regularized Wasserstein Gradient Flow), which rigorously formalizes posterior sampling with latent diffusion models as a regularized gradient flow of KL divergence in the Wasserstein-2 space. An ODE system in the latent space is derived to solve image inverse problems, achieving substantially higher PSNR than baselines on inpainting and super-resolution tasks on FFHQ-512.

## Background & Motivation

**Background**: Image inverse problems (inpainting, super-resolution, deblurring, etc.) require strong priors. Diffusion models have recently emerged as state-of-the-art priors, guiding posterior distributions by modifying the sampling process. Common approaches include DPS (gradient guidance), RED-Diff (variational inference), and SMC-based methods.

**Limitations of Prior Work**: Most methods are designed for pixel-space diffusion models, whereas computationally more efficient latent diffusion models (e.g., Stable Diffusion) define priors over latent variables $z_0$ rather than $x_0$. Adapting pixel-space methods to the latent space is non-trivial. Existing latent-space methods (PSLD, RLSD) either produce suboptimal image quality or lack rigorous theoretical foundations.

**Key Challenge**: The prior of a latent diffusion model (LDM) is defined over $z_0$, while the observation likelihood is defined over $x_0$; these two spaces are connected through a nonlinear encoder-decoder, making principled posterior sampling in the latent space nontrivial.

**Goal**
- How can one rigorously derive, from first principles, a method for solving inverse problems using latent diffusion models?
- How can both the data likelihood and the diffusion prior be jointly considered in the latent space?

**Key Insight**: Posterior sampling is framed as an optimization problem over the space of probability measures — minimizing the KL divergence between the approximate and true posteriors in the Wasserstein-2 space, augmented with a diffusion-model-based regularization term.

**Core Idea**: Posterior sampling in the latent space is formalized as a regularized Wasserstein gradient flow, from which a particle ODE system is derived and simulated using the Adam optimizer.

## Method

### Overall Architecture
The DWGF framework maintains $N$ particles $\{z_0^{(i)}\}$ in the latent space $\mathcal{Z}$ and iteratively updates their positions by simulating a Wasserstein gradient flow ODE, driving the particle distribution toward the posterior $\mu(z_0|y)$. The objective has two terms: (1) the KL divergence term $\mathcal{F}[\mu]$ — measuring the discrepancy between the distribution mapped to pixel space via the decoder and the true posterior; and (2) the diffusion prior regularization term $\mathcal{R}[\mu]$ — enforcing alignment with the pretrained diffusion model's marginals across all noise levels.

### Key Designs

1. **Derivation of the KL Divergence Gradient Flow ($\nabla \delta\mathcal{F}/\delta\mu$)**

    - **Function**: Derives the gradient of the pixel-space posterior constraint in the latent space.
    - **Mechanism**: The objective is to minimize $D_{\text{KL}}(q_\mu(x_0|y) \| p(x_0|y))$, where $q_\mu(x_0|y) = \int p_{\phi^-}(x_0|z_0) \mu(z_0|y) dz_0$. Via first variation and the reparameterization trick, the gradient simplifies to: $\nabla_{z_0} \frac{\delta\mathcal{F}}{\delta\mu} = -\mathbb{E}_\epsilon[\nabla_{x_0} \log p(x_0|y) \cdot \frac{\partial \mathcal{D}_{\phi^-}(z_0)}{\partial z_0}]$. The posterior score decomposes into a likelihood score (analytically tractable under Gaussianity) and a data prior score (approximated via the VAE encoder).
    - **Design Motivation**: Grounding the derivation in KL divergence ensures theoretical correctness — the gradient flow converges to the distribution minimizing the information divergence. The chain rule through the VAE encoder-decoder naturally propagates gradients between the latent and pixel spaces.

2. **Weighted KL Divergence Regularization ($\mathcal{R}[\mu]$)**

    - **Function**: Employs the pretrained diffusion model as a latent-space prior regularizer.
    - **Mechanism**: $\mathcal{R}(\mu) = \int_0^T w(s) D_{\text{KL}}(\mu(z_s|y) \| p_{\theta^-}(z_s)) ds$ enforces that the particle distribution aligns with the diffusion model's marginals at all noise levels $s$. The distribution of $z_0$ is pushed forward to each noise level via the forward kernel $p(z_s|z_0)$ and compared against the diffusion model's score at that level. The gradient is: $\nabla_{z_0} \frac{\delta\mathcal{R}}{\delta\mu} = \mathbb{E}_{s,\epsilon}[\tilde{w}(s)(\nabla_{z_s} \log \int p(z_s|z_0) \mu(z_0|y) dz_0 - \nabla_{z_s} \log p_{\theta^-}(z_s)) \cdot \alpha_s]$.
    - **Design Motivation**: The weighted KL divergence is a theoretically well-behaved regularizer — non-negative, convex, and minimized at the same point as the standard KL divergence (Theorem 2.1). Multi-scale regularization across all noise levels provides multi-resolution prior constraints.

3. **Particle ODE System + Adam Optimizer**

    - **Function**: Approximates the gradient flow with $N$ particles and solves it efficiently.
    - **Mechanism**: The ODE $\frac{dz_{0,t}}{dt} = -(\nabla_{z_0} \frac{\delta\mathcal{F}}{\delta\mu} + \gamma \nabla_{z_0} \frac{\delta\mathcal{R}}{\delta\mu})$ is integrated per particle using Monte Carlo approximation. Rather than simple Euler steps, the Adam optimizer is used to simulate the flow — treating the gradient flow drift as a gradient and exploiting Adam's adaptive learning rate to accelerate convergence.
    - **Design Motivation**: The optimization landscape of inverse problems is complex, and naive Euler steps converge slowly; Adam's momentum and adaptive step size provide significant acceleration.

### Loss & Training
- **Training-free**: Directly uses pretrained Stable Diffusion without any additional training or fine-tuning.
- **Particle count $N$**: Multiple particles evolve simultaneously; multi-particle interaction terms are approximated via Monte Carlo.
- The regularization strength $\gamma$ and weight function $w(s)$ are hyperparameters.

## Key Experimental Results

### Main Results: FFHQ-512

| Method | Task | FID↓ | PSNR↑ | LPIPS↓ |
|--------|------|------|-------|--------|
| PSLD | Inpainting (Box) | 57.70 | 22.72 | 0.082 |
| RLSD | Inpainting (Box) | **29.18** | 24.98 | **0.079** |
| **DWGF** | Inpainting (Box) | 118.05 | **27.56** | 0.184 |
| PSLD | SR ×8 | 81.31 | 24.82 | 0.314 |
| RLSD | SR ×8 | **65.42** | 28.39 | 0.286 |
| **DWGF** | SR ×8 | 101.94 | **32.71** | **0.193** |

### Ablation Study

| Metric | DWGF vs. RLSD |
|--------|---------------|
| PSNR | DWGF leads by a large margin (inpainting +2.58, SR +4.32) |
| FID | RLSD leads by a large margin (DWGF FID is substantially worse) |
| LPIPS | Task-dependent (DWGF better on SR; RLSD better on inpainting) |

### Key Findings
- **High PSNR but poor FID**: DWGF excels in pixel-level fidelity but produces images that are less perceptually realistic overall — a consequence of the mode-seeking nature of reverse KL divergence, which tends to find a single posterior mode rather than covering the full distribution, resulting in slightly blurry reconstructions.
- **Effective diversity via multiple particles**: Figure 1(a) shows that multiple particles yield diverse yet plausible reconstructions, confirming that the method performs sampling rather than point optimization.
- **Deterministic encoding approximation is valid**: The log-variance of the VAE encoder is approximately $-17$, close to a deterministic mapping, which simplifies the computation of the data score term.

## Highlights & Insights
- **Rigorous first-principles derivation**: Rather than heuristically modifying the sampling process, the ODE system is derived from the Wasserstein gradient flow framework, providing a solid theoretical foundation.
- **Theoretical guarantees of weighted KL regularization** (non-negativity, convexity, shared minimizer with standard KL) ensure well-behaved regularization properties.
- **Training-free**: Directly leverages pretrained Stable Diffusion without any fine-tuning or additional training.
- **Gradient propagation between latent and pixel spaces** is naturally achieved via the Jacobian of the encoder-decoder, avoiding the ad-hoc approximations present in prior work.

## Limitations & Future Work
- **Poor FID**: The mode-seeking nature of reverse KL divergence leads to blurry reconstructions — the authors suggest adding entropy regularization or repulsive potentials as remedies.
- **Only two tasks evaluated**: Experiments are limited to inpainting and super-resolution; common inverse problems such as denoising, deblurring, and compressed sensing are not evaluated.
- **Computational cost not reported**: The overhead of multi-particle ODE simulation combined with Adam optimization steps is not discussed.
- **Single dataset (FFHQ)**: Generalization to other datasets such as ImageNet or LSUN is not validated.
- **Mode-seeking vs. mode-covering**: Reverse KL is inherently mode-seeking; switching to forward KL, MMD, or alternative divergences may be necessary.

## Related Work & Insights
- **vs. PSLD (Rout et al.)**: PSLD adapts DPS to the latent space with a data consistency regularizer; DWGF is more theoretically grounded via Wasserstein gradient flow, achieving higher PSNR at the cost of worse FID.
- **vs. RLSD (Zilberstein et al.)**: RLSD performs particle simulation in the joint pixel-latent product space with repulsive potentials to prevent mode collapse; DWGF operates purely in the latent space but lacks repulsive potentials, leading to worse FID.
- **vs. Wang et al. (2023)**: DWGF shares the same theoretical framework (Wasserstein gradient flow of KL divergence) as Wang et al., but extends it to latent diffusion models.
- **Interesting connection**: Under deterministic encoding, the data score term closely resembles the data consistency terms in PSLD and ReSample — suggesting that these heuristic methods can in fact be derived from the gradient flow framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying the Wasserstein gradient flow framework to inverse problems with latent diffusion models is novel, though the core mathematical tools are drawn from existing literature.
- **Experimental Thoroughness**: ⭐⭐⭐ Only two tasks and one dataset are evaluated; poor FID performance limits the overall picture.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and clear, but the experimental section is too brief.
- **Value**: ⭐⭐⭐ The theoretical contribution is meaningful, but the experimental performance (FID) is not yet convincing; further improvement is needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[NeurIPS 2025\] Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)
- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)

</div>

<!-- RELATED:END -->
