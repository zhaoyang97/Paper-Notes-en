---
title: >-
  [Paper Note] Learning Normalized Energy Models for Linear Inverse Problems
description: >-
  [ICML 2026][Image Restoration][Paper Note] The authors reformulate "linear inverse problems" as "anisotropic denoising" and propose Anisotropic Covariance Score Matching (A-CSM) to train a **normalized** energy model $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$. A single model can handle inpainting, deblurring, and s
tags:
  - ICML 2026
  - Image Restoration
date: 2026-05-08
content_hash: 7f864a8bad70c1c9
---
# Learning Normalized Energy Models for Linear Inverse Problems

**Conference**: ICML 2026  
**arXiv**: [2605.15487](https://arxiv.org/abs/2605.15487)  
**Code**: https://github.com/nzilberstein/Anisotropic-energy-Model (Yes)  
**Area**: Image Restoration / Energy Models / Diffusion Models / Linear Inverse Problems  
**Keywords**: Anisotropic Denoising, Covariance Score Matching, Normalized Energy Models, Posterior Sampling, Blind Inverse Problems

## TL;DR
The authors reformulate "linear inverse problems" as "anisotropic denoising" and propose Anisotropic Covariance Score Matching (A-CSM) to train a **normalized** energy model $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$. A single model can handle inpainting, deblurring, and super-resolution, while unlocking three new capabilities: energy-guided adaptive scheduling, MALA unbiased correction, and blind inverse problem estimation.

## Background & Motivation

**Background**: Diffusion models have become the dominant prior for image inverse problems (deblurring, completion, super-resolution). Current approaches generally fall into two categories: the Bayesian camp, which treats a pre-trained unconditional diffusion model as $p(\mathbf{x})$ and uses Bayes' rule to compute $\nabla\log p(\mathbf{x}_t|\mathbf{y})$ during sampling; and the Regression camp, which directly learns the conditional score $\nabla\log p(\mathbf{x}_t|\mathbf{y})$ using paired $(\mathbf{x},\mathbf{y})$ data.

**Limitations of Prior Work**: In the Bayesian camp, the likelihood term $p(\mathbf{y}|\mathbf{x}_t)=\int p(\mathbf{y}|\mathbf{x})p(\mathbf{x}|\mathbf{x}_t)\mathrm{d}\mathbf{x}$ is a high-dimensional integral that must be approximated using methods like DPS, which introduces sampling bias. While the Regression camp avoids these approximations, a new model must be retrained for every different degradation operator $\mathbf{H}$, losing the flexibility of prior/likelihood decoupling. More fundamentally, both categories are score-based, learning only the gradient rather than the **density itself**. Consequently, they cannot perform normalized log-probability comparisons, MCMC acceptance probability calculations, energy-guided scheduling, or blind estimation tasks such as $\arg\max_{\boldsymbol{\Sigma}} p(\mathbf{y}|\boldsymbol{\Sigma})$.

**Key Challenge**: The goal is to simultaneously achieve (i) prior flexibility shared across degradations, (ii) unbiased sampling without relying on likelihood approximations, and (iii) explicit normalized densities. Existing EBM-with-diffusion studies (Du 2023, Thornton 2025) only support isotropic noise, which cannot cover the **anisotropic** covariance inherent in linear inverse problems.

**Key Insight**: The authors observe that $\mathbf{y}=\mathbf{H}\mathbf{x}+\sigma\mathbf{v}$, when rewritten via $\mathbf{H}^{-1}\mathbf{y}$, is equivalent to $\mathbf{y}=\mathbf{x}+\boldsymbol{\Sigma}^{1/2}\mathbf{v}'$, where $\boldsymbol{\Sigma}=\sigma^2\mathbf{H}^{-1}(\mathbf{H}^{-1})^\top$. Thus, "solving a family of linear inverse problems" is equivalent to "denoising over a family of covariances $\boldsymbol{\Sigma}$." By learning a density conditioned on $\boldsymbol{\Sigma}$, all linear degradations can be unified.

**Core Idea**: The dual score matching of Guth 2025 is extended from isotropic to anisotropic, introducing a **covariance score** term $\nabla_{\boldsymbol{\Sigma}}U_\theta$. This term is constrained by the Fokker-Planck equation to ensure mass conservation across $\boldsymbol{\Sigma}$, thereby training the "unnormalized energy" into a "normalized energy."

## Method

### Overall Architecture
The method aims to unifiedly handle a family of linear inverse problems using a single model. The key transformation involves rewriting the degraded observation $\mathbf{y}=\mathbf{H}\mathbf{x}+\sigma\mathbf{v}$ as an anisotropic denoising problem $\mathbf{y}=\mathbf{x}+\boldsymbol{\Sigma}^{1/2}\mathbf{v}'$. Consequently, it suffices to learn a normalized energy $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$ conditioned on the noise covariance $\boldsymbol{\Sigma}$. Taking the gradient with respect to $\mathbf{y}$ yields the score $\nabla_\mathbf{y}U_\theta$, which is used for reconstruction via the anisotropic Tweedie formula $\mathbb{E}[\mathbf{x}|\mathbf{y},\boldsymbol{\Sigma}]=\mathbf{y}-\boldsymbol{\Sigma}\nabla_\mathbf{y}U_\theta$. The gradient with respect to $\boldsymbol{\Sigma}$ provides the covariance score for adaptive scheduling and blind estimation. Architecturally, the energy is defined as $U_\theta(\mathbf{y},\boldsymbol{\Sigma})=\tfrac{1}{2}\langle\mathbf{y},\mathbf{s}_\theta(\mathbf{y},\boldsymbol{\Sigma})\rangle$, with an EDM (Karras 2022) UNet as the backbone.

### Key Designs

**1. Anisotropic Denoising Score Matching (A-DSM): Training Energy Models as Covariance-Aware Denoisers**

Standard DSM assumes $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$. When encountering covariances like inpainting (where noise magnitudes along different directions differ by orders of magnitude), gradient scales can explode or collapse. A-DSM uses the anisotropic Tweedie formula to approximate the energy as a denoiser. The loss is defined as $\ell_{\text{A-DSM}}=\mathbb{E}[\|\boldsymbol{\Sigma}^{1/2}\nabla_\mathbf{y}U_\theta(\mathbf{y},\boldsymbol{\Sigma})-\boldsymbol{\Sigma}^{-1/2}(\mathbf{y}-\mathbf{x})\|^2]$, where both sides are reweighted by $\boldsymbol{\Sigma}^{1/2}$ to ensure scale invariance. This is an anisotropic generalization of maximum-likelihood weighting. This scale-invariant reweighting allows $\nabla_\mathbf{y}U_\theta$ to stably approximate $\nabla_\mathbf{y}\log p(\mathbf{y}|\boldsymbol{\Sigma})$ over an extremely wide range of noise variances $[10^{-9},10^3]$, which is a prerequisite for any-order generation and blind estimation.

**2. Anisotropic Covariance Score Matching (A-CSM): Training Normalized Energy via Fokker-Planck Constraints**

A-DSM only learns gradients in the $\mathbf{y}$ direction, leaving energy values across different $\boldsymbol{\Sigma}$ separated by an unknown constant—a limitation of isotropic versions (Guth 2025, Yu 2025). A-CSM additionally supervises the energy gradient with respect to $\boldsymbol{\Sigma}$. The authors prove a covariance Tweedie identity $\nabla_{\boldsymbol{\Sigma}}U(\mathbf{y},\boldsymbol{\Sigma})=\mathbb{E}[\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}-\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}(\mathbf{y}-\mathbf{x})(\mathbf{y}-\mathbf{x})^\top\boldsymbol{\Sigma}^{-1}]$ and apply a scale-invariant loss $\ell_{\text{A-CSM}}$ under the Frobenius norm. The total objective is $\tfrac{1}{d}\ell_{\text{A-DSM}}+\tfrac{1}{d^2}\ell_{\text{A-CSM}}$. Essentially, the Fokker-Planck continuity equation $\nabla_{\boldsymbol{\Sigma}}p(\mathbf{y}|\boldsymbol{\Sigma})=\tfrac{1}{2}\nabla_\mathbf{y}^2 p(\mathbf{y}|\boldsymbol{\Sigma})$ is used as a hard constraint to maintain marginal density consistency, ensuring the normalization constant does not depend on $\boldsymbol{\Sigma}$. After training, $U_\theta(\mathbf{y},\boldsymbol{\Sigma})-\mathbb{E}_\mathbf{y}[U_\theta|\boldsymbol{\Sigma}]+\tfrac{1}{2}\log\det(2\pi e\boldsymbol{\Sigma})$ is used for renormalization (with $\mathbf{y}\sim\mathcal{N}(0,\boldsymbol{\Sigma})$ as an anchor for large $\boldsymbol{\Sigma}$). This step enables the model to calculate probability ratios like $p(\mathbf{y}|\boldsymbol{\Sigma}_T,\mathbf{y})/p(\mathbf{y}|\boldsymbol{\Sigma}_t,\mathbf{y})$, perform MALA acceptance, and solve $\arg\max_{\boldsymbol{\Sigma}}\log p(\mathbf{y}|\boldsymbol{\Sigma})$, fundamentally distinguishing it from all pure score models.

**3. Dual-Domain Covariance Embedding: A Single UNet for Pixel and Frequency Covariances**

A fully general $\boldsymbol{\Sigma}$ has $d(d-1)/2$ degrees of freedom, which leads to memory explosion at $d=64^2=4096$. However, spatial-diagonal matrices cover inpainting, and frequency-diagonal matrices cover deblurring/SR. Limiting the conditions to these families (compressed to $d$-dimensional vectors) does not sacrifice practical expressivity. Spatial covariance is represented as a spatially varying noise map $\mathbf{e}_\ell\in\mathbb{R}^{c_\ell\times d_\ell}$, and frequency covariance as a channel-only modulation $\mathbf{e}_\ell\in\mathbb{R}^{c_\ell}$. Embeddings for both branches are calculated in parallel and injected into each layer via $\mathbf{x}_\ell\leftarrow\mathrm{SiLU}(\mathbf{x}_\ell\odot(1+\mathbf{e}_\ell))$. This modulation is fully compatible with EDM’s native isotropic channel gain modulation, adding almost no computational overhead and allowing the model to inherit denoising inductive biases from existing score architectures.

### Loss & Training
The total loss is $\mathcal{L}=\tfrac{1}{d}\ell_{\text{A-DSM}}+\tfrac{1}{d^2}\ell_{\text{A-CSM}}$. Training covariances are sampled with a 0.5/0.5 ratio between spatial classes (central/lateral boxes, size 1–64) and spectral classes (Gaussian deblur kernel size 8×8, $\sigma_g=0.8$; 4× SR). An EDM UNet is used as the backbone. All methods (including baselines) use the same architecture, differing only in $p(\boldsymbol{\Sigma})$ and inputs: the Bayesian baseline sets $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$, while Palette stacks the measurement and noise map as input. Sampling uses a maximum of 1000 NFE (1200 for CelebA inpainting).

## Key Experimental Results

### Main Results
Inpainting (center 45×45 box, $\sigma=10^{-4}$) and Gaussian deblurring (8×8 kernel, $\sigma=10^{-2}$) tasks on CelebA 64×64 and ImageNet 64×64, comparing against DPS, RED-Diff, DAPS, and Palette.

| Dataset / Task | Metric | Ours | DPS | RED-Diff | DAPS |
|----------------|--------|------|-----|----------|------|
| CelebA Inpainting | LPIPS↓ | **0.093** | 0.110 | 0.100 | 0.098 |
| CelebA Inpainting | FID↓ | **34.57** | 36.76 | 47.82 | 45.76 |
| CelebA Deblurring | LPIPS↓ | **0.002** | 0.004 | 0.006 | 0.005 |
| CelebA Deblurring | DISTS↓ | **0.04** | 0.08 | 0.08 | 0.10 |
| ImageNet Inpainting | FID↓ | **47.54** | 55.61 | 58.50 | 54.07 |
| ImageNet Deblurring | FID↓ | **44.82** | 59.09 | 63.10 | 79.43 |
| ImageNet Deblurring | DISTS↓ | **0.07** | 0.10 | 0.11 | 0.15 |

In terms of PSNR, RED-Diff is slightly higher for CelebA inpainting (17.96 vs 17.70), consistent with its over-smoothing MAP behavior. However, Ours wins almost entirely in perceptual quality (LPIPS/FID/DISTS), especially in the deblurring task where DISTS is halved.

### Ablation Study
ULA vs MALA correction steps for the CelebA inpainting task (LPIPS↓):

| Corrector | 1 Step | 5 Steps | 8 Steps |
|-----------|--------|---------|---------|
| ULA | 0.093 | 0.093 | 0.093 |
| MALA | 0.093 | 0.091 | **0.089** |

A-CSM ablation is detailed in §4.3 for blind tasks: an A-DSM-only model without A-CSM completely fails to locate the box size and $\sigma_1$, verifying that normalization consistency across $\boldsymbol{\Sigma}$ is the root cause for successful blind estimation.

### Key Findings
- **Alignment of Energy and Sample Quality**: Using the same $U_\theta$ to evaluate samples from DPS, RED-Diff, and Ours—DPS samples show significantly lower prior probability than GT (likelihood approximation biases them toward OOD regions); RED-Diff samples show high prior probability but low posterior probability (over-smoothed). Only samples from Ours are close to GT in both prior and posterior, making "prior/posterior log-probability" a computable diagnostic metric for inverse problem samplers.
- **Benefits of Energy-Guided Adaptive Scheduling for Low Measurement Regimes**: On MNIST for random $k$-pixel reconstruction, energy-guided scheduling $\boldsymbol{\delta}\boldsymbol{\Sigma}_t\propto\boldsymbol{\Sigma}_t\nabla_{\boldsymbol{\Sigma}}U_\theta\boldsymbol{\Sigma}_t$ (steepest descent under Bregman geometry) consistently yields lower classification error rates than geometric scheduling when $k$ is small.
- **Unbiased MALA Correction Requires Normalized Energy Models**: MALA acceptance requires computing $p(\mathbf{x}'|\boldsymbol{\Sigma}_T,\mathbf{y})/p(\mathbf{x}|\boldsymbol{\Sigma}_T,\mathbf{y})$, which pure score models cannot perform. Ours uses 8 MALA steps to reduce LPIPS from 0.093 to 0.089, whereas increasing ULA steps yields no improvement, indicating this is an "irreplaceable capability."
- **Blind Estimation via $\arg\max_{\boldsymbol{\Sigma}}\log p_\theta(\mathbf{y}|\boldsymbol{\Sigma})$**: In inpainting tasks with unknown box-size and $\sigma_1$, the log-probability surface exhibits a clear unimodal peak at the ground truth. This is impossible for Bayesian methods (which lack $\log p(\mathbf{y}|\boldsymbol{\Sigma})$) and Regression methods (which require retraining for each $\boldsymbol{\Sigma}$).

## Highlights & Insights
- **Perspective Shift of "Linear Degradation ≡ Anisotropic Noise"**: Rewriting the problem via $\mathbf{H}^{-1}\mathbf{y}$ unifies various inverse problems into a single denoising framework, reducing the task to modeling a family of $\boldsymbol{\Sigma}$.
- **A-CSM as Implicit Fokker-Planck Enforcement**: Turning the hard physical constraint of "normalization constant invariance across $\boldsymbol{\Sigma}$" into a differentiable training loss. This "regularization via conservation equations" can be transferred to any conditional density modeling.
- **Energy Values as Evaluation Metrics**: Rather than relying solely on PSNR/LPIPS, $\log p(\hat{\mathbf{x}})$ histograms can now diagnose whether a sampler is biased toward OOD regions or over-smoothed.
- **Diagonal Limitation ≠ Poor Expressivity**: Selecting spatial-diagonal and spectral-diagonal families covers over 90% of common linear inverse problems, efficiently compressing $O(d^2)$ degrees of freedom to $O(d)$.

## Limitations & Future Work
- The authors admit that dual score matching is more expensive than pure score models (requiring extra backprop to compute $\nabla_{\boldsymbol{\Sigma}}U_\theta$). Future work could use sliced score matching + forward-mode JVP for acceleration.
- Covariances are restricted to spatial/spectral diagonals, failing to represent arbitrary $\boldsymbol{\Sigma}$ (e.g., rotational blur or spatially varying blur).
- Resolution is currently limited to 192×192 (AFHQ-Cat); scaling to 256+ or 1024 remains unproven, and EBM training stability may be at risk in higher dimensions.
- The weighting of $1/d$ and $1/d^2$ in the objective lacks comprehensive ablation.
- Adding new degradation operators still requires expansion training to adjust the range of $\log p(\mathbf{y}|\boldsymbol{\Sigma})$.

## Related Work & Insights
- **vs DPS / DAPS (Bayesian)**: These rely on Gaussian approximations for $p(\mathbf{y}|\mathbf{x}_t)$. Ours directly learns normalized $p(\mathbf{y}|\boldsymbol{\Sigma})$, avoiding OOD bias at the cost of needing the covariance family $p(\boldsymbol{\Sigma})$ beforehand.
- **vs Palette / InDI (Regression)**: These train a separate model for each degradation. Ours uses a single model to cover a family of $\boldsymbol{\Sigma}$ and supports blind estimation. 
- **vs Guth 2025 / Yu 2025 (Isotropic EBM)**: Their models only normalize on the $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$ manifold. This work provides the "isotropic → anisotropic" jump necessary for inverse problems.
- **vs Du 2023 / Thornton 2025 (Compositional EBM)**: While they focus on compositional generation, this work emphasizes "normalization," enabling MALA, blind estimation, and probability comparisons.
- **Insight**: Reformulating "inverse problems" as "conditional density family modeling" plus "Fokker-Planck constraints" is a recipe that could be ported to non-linear problems, cross-modal generation, or scientific computing inversion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PnP-CM: Consistency Models as Plug-and-Play Priors for Inverse Problems](../../CVPR2026/image_restoration/pnp-cm_consistency_models_as_plug-and-play_priors_for_inverse_problems.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[CVPR 2026\] Variational Garrote for Sparse Inverse Problems](../../CVPR2026/image_restoration/variational_garrote_for_sparse_inverse_problems.md)
- [\[CVPR 2026\] Outlier-Robust Diffusion Solvers for Inverse Problems](../../CVPR2026/image_restoration/outlier-robust_diffusion_solvers_for_inverse_problems.md)
- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](../../NeurIPS2025/image_restoration/learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)

</div>

<!-- RELATED:END -->
