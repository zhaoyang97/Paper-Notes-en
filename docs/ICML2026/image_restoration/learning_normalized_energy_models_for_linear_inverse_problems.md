---
title: >-
  [Paper Note] Learning Normalized Energy Models for Linear Inverse Problems
description: >-
  [ICML 2026][Image Restoration][Paper Note] The authors reformulate "linear inverse problems" as "anisotropic denoising" and propose Anisotropic Covariance Score Matching (A-CSM) to train a **normalized** energy model $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$. A single model can handle inpainting, deblurring, and s
tags:
  - ICML 2026
  - Image Restoration
date: 2026-05-08
content_hash: 665c1606d78656e6
---
# Learning Normalized Energy Models for Linear Inverse Problems

**Conference**: ICML 2026  
**arXiv**: [2605.15487](https://arxiv.org/abs/2605.15487)  
**Code**: https://github.com/nzilberstein/Anisotropic-energy-Model (Available)  
**Area**: Image Restoration / Energy-Based Models / Diffusion Models / Linear Inverse Problems  
**Keywords**: Anisotropic Denoising, Covariance Score Matching, Normalized Energy Models, Posterior Sampling, Blind Inverse Problems

## TL;DR
The authors reformulate "linear inverse problems" as "anisotropic denoising" and propose Anisotropic Covariance Score Matching (A-CSM) to train a **normalized** energy model $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$. A single model can handle inpainting, deblurring, and super-resolution while unlocking three new capabilities: energy-guided adaptive scheduling, MALA unbiased correction, and blind inverse estimation.

## Background & Motivation

**Background**: Diffusion models have become the mainstream prior for image inverse problems (deblurring, inpainting, super-resolution). Current approaches generally fall into two categories: the Bayesian school, which treats a pre-trained unconditional diffusion model as $p(\mathbf{x})$ and approximates $\nabla\log p(\mathbf{x}_t|\mathbf{y})$ via Bayes' rule during sampling; and the Regression school, which directly learns the conditional score $\nabla\log p(\mathbf{x}_t|\mathbf{y})$ using $(\mathbf{x}, \mathbf{y})$ pairs.

**Limitations of Prior Work**: For the Bayesian school, the likelihood term $p(\mathbf{y}|\mathbf{x}_t)=\int p(\mathbf{y}|\mathbf{x})p(\mathbf{x}|\mathbf{x}_t)\mathrm{d}\mathbf{x}$ involves high-dimensional integration, necessitating approximations like DPS that introduce sampling bias. While the Regression school avoids such approximations, it requires retraining a model for every different degradation operator $\mathbf{H}$, losing the flexibility of decoupled priors and likelihoods. A more fundamental issue is that both are score-based, learning only gradients and not the **density itself**, making them incapable of normalized log-probability comparison for MCMC acceptance, energy-guided scheduling, or blind estimation tasks like $\arg\max_{\boldsymbol{\Sigma}} p(\mathbf{y}|\boldsymbol{\Sigma})$.

**Key Challenge**: To simultaneously achieve: (i) cross-degradation prior flexibility, (ii) unbiased sampling without likelihood approximation, and (iii) explicit normalized density. Existing EBM-with-diffusion works (Du 2023, Thornton 2025) only support isotropic noise, which cannot cover the **anisotropic** covariance inherent in linear inverse problems.

**Key Insight**: The authors observe that $\mathbf{y}=\mathbf{H}\mathbf{x}+\sigma\mathbf{v}$, when rewritten via $\mathbf{H}^{-1}\mathbf{y}$, is equivalent to $\mathbf{y}=\mathbf{x}+\boldsymbol{\Sigma}^{1/2}\mathbf{v}'$, where $\boldsymbol{\Sigma}=\sigma^2\mathbf{H}^{-1}(\mathbf{H}^{-1})^\top$. Thus, "solving a family of linear inverse problems" is equivalent to "denoising over a family of covariances $\boldsymbol{\Sigma}$." By learning a density conditioned on $\boldsymbol{\Sigma}$, all linear degradations can be unified.

**Core Idea**: Generalize the dual score matching of Guth 2025 from isotropic to anisotropic cases by introducing a **covariance score** term $\nabla_{\boldsymbol{\Sigma}}U_\theta$. This term is constrained by the Fokker-Planck equation to ensure mass conservation across $\boldsymbol{\Sigma}$, effectively training an "unnormalized energy" into a "normalized energy".

## Method

### Overall Architecture
The goal is to handle a family of linear inverse problems with a single model. The key transformation rewrites the degraded observation $\mathbf{y}=\mathbf{H}\mathbf{x}+\sigma\mathbf{v}$ as an anisotropic denoising problem $\mathbf{y}=\mathbf{x}+\boldsymbol{\Sigma}^{1/2}\mathbf{v}'$. Consequently, one only needs to learn a normalized energy $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$ conditioned on the noise covariance $\boldsymbol{\Sigma}$. Taking the gradient w.r.t. $\mathbf{y}$ yields the score $\nabla_\mathbf{y}U_\theta$, used for reconstruction via the anisotropic Tweedie formula $\mathbb{E}[\mathbf{x}|\mathbf{y},\boldsymbol{\Sigma}]=\mathbf{y}-\boldsymbol{\Sigma}\nabla_\mathbf{y}U_\theta$. Taking the gradient w.r.t. $\boldsymbol{\Sigma}$ yields the covariance score for adaptive scheduling and blind estimation. The architecture represents energy as $U_\theta(\mathbf{y},\boldsymbol{\Sigma})=\tfrac{1}{2}\langle\mathbf{y},\mathbf{s}_\theta(\mathbf{y},\boldsymbol{\Sigma})\rangle$, using an EDM (Karras 2022) UNet as the backbone.

### Key Designs

**1. Anisotropic Denoising Score Matching (A-DSM): Training Energy Models as Covariance-Aware Denoisers**

Standard DSM assumes $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$. In tasks like inpainting, where noise magnitudes along different directions vary by orders of magnitude, gradient scales can explode or collapse. A-DSM uses the anisotropic Tweedie formula to approximate the energy as a denoiser, with a loss $\ell_{\text{A-DSM}}=\mathbb{E}[\|\boldsymbol{\Sigma}^{1/2}\nabla_\mathbf{y}U_\theta(\mathbf{y},\boldsymbol{\Sigma})-\boldsymbol{\Sigma}^{-1/2}(\mathbf{y}-\mathbf{x})\|^2]$. Reweighting both sides with $\boldsymbol{\Sigma}^{1/2}$ makes the loss scale-invariant, serving as an anisotropic generalization of maximum-likelihood weighting. This allows $\nabla_\mathbf{y}U_\theta$ to stably approximate $\nabla_\mathbf{y}\log p(\mathbf{y}|\boldsymbol{\Sigma})$ across a wide range of noise variances $[10^{-9},10^3]$, enabling any-order generation and blind estimation.

**2. Anisotropic Covariance Score Matching (A-CSM): Normalizing Energy via Fokker-Planck Constraints**

A-DSM only learns the gradient in the $\mathbf{y}$ direction; energy values across different $\boldsymbol{\Sigma}$ remain offset by an unknown constant, preventing log-probability comparisons—a limitation of isotropic versions (Guth 2025, Yu 2025). A-CSM additionally supervises the gradient of energy w.r.t. $\boldsymbol{\Sigma}$. By proving a covariance-version Tweedie identity $\nabla_{\boldsymbol{\Sigma}}U(\mathbf{y},\boldsymbol{\Sigma})=\mathbb{E}[\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}-\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}(\mathbf{y}-\mathbf{x})(\mathbf{y}-\mathbf{x})^\top\boldsymbol{\Sigma}^{-1}]$ and applying a scale-invariant Frobenius norm loss $\ell_{\text{A-CSM}}$, the overall objective becomes $\tfrac{1}{d}\ell_{\text{A-DSM}}+\tfrac{1}{d^2}\ell_{\text{A-CSM}}$. Essentially, this uses the Fokker-Planck continuity equation $\nabla_{\boldsymbol{\Sigma}}p(\mathbf{y}|\boldsymbol{\Sigma})=\tfrac{1}{2}\nabla_\mathbf{y}^2 p(\mathbf{y}|\boldsymbol{\Sigma})$ to enforce consistency across all marginal densities, making the normalization constant independent of $\boldsymbol{\Sigma}$. After training, renormalization is performed using $U_\theta(\mathbf{y},\boldsymbol{\Sigma})-\mathbb{E}_\mathbf{y}[U_\theta|\boldsymbol{\Sigma}]+\tfrac{1}{2}\log\det(2\pi e\boldsymbol{\Sigma})$ (with $\mathbf{y}\sim\mathcal{N}(0,\boldsymbol{\Sigma})$ at large $\boldsymbol{\Sigma}$ as an anchor). This step allows the calculation of probability ratios, MALA acceptance, and $\arg\max_{\boldsymbol{\Sigma}}\log p(\mathbf{y}|\boldsymbol{\Sigma})$, distinguishing this approach from pure score-based models.

**3. Dual-Domain Covariance Embedding: A UNet for Spatial and Spectral Covariances**

A fully general $\boldsymbol{\Sigma}$ has $d(d-1)/2$ degrees of freedom, causing memory explosion at $d=64^2$. However, spatial-diagonal matrices cover inpainting, and frequency-diagonal matrices cover deblurring/SR. By restricting $\boldsymbol{\Sigma}$ to these two families and compressing them into $d$-dimensional vectors, expressive power is maintained. Spatial covariance is represented as a spatially-varying noise map $\mathbf{e}_\ell\in\mathbb{R}^{c_\ell\times d_\ell}$, while spectral covariance is represented as channel-only modulation $\mathbf{e}_\ell\in\mathbb{R}^{c_\ell}$. Embeddings from both branches are injected into each layer via $\mathbf{x}_\ell\leftarrow\mathrm{SiLU}(\mathbf{x}_\ell\odot(1+\mathbf{e}_\ell))$. This modulation is compatible with the native isotropic gain modulation in EDM, adding negligible computational overhead while inheriting existing inductive biases for denoising.

### Loss & Training
Total loss $\mathcal{L}=\tfrac{1}{d}\ell_{\text{A-DSM}}+\tfrac{1}{d^2}\ell_{\text{A-CSM}}$. Training covariances are sampled equally (0.5/0.5) between spatial types (central/lateral boxes, size 1–64) and spectral types (Gaussian deblur kernel size 8×8, $\sigma_g=0.8$; 4× SR). The backbone is an EDM UNet. All baselines use the same architecture: Bayesian baselines set $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$, while Palette stacks measurement and noise as input. Sampling uses up to 1000 NFE (1200 for CelebA inpainting).

## Key Experimental Results

### Main Results
Inpainting (central 45×45 box, $\sigma=10^{-4}$) and Gaussian deblurring (8×8 kernel, $\sigma=10^{-2}$) on CelebA 64×64 and ImageNet 64×64, compared against DPS, RED-Diff, DAPS, and Palette.

| Dataset / Task | Metric | Ours | DPS | RED-Diff | DAPS |
|----------------|--------|------|-----|----------|------|
| CelebA Inpainting | LPIPS↓ | **0.093** | 0.110 | 0.100 | 0.098 |
| CelebA Inpainting | FID↓ | **34.57** | 36.76 | 47.82 | 45.76 |
| CelebA Deblurring | LPIPS↓ | **0.002** | 0.004 | 0.006 | 0.005 |
| CelebA Deblurring | DISTS↓ | **0.04** | 0.08 | 0.08 | 0.10 |
| ImageNet Inpainting | FID↓ | **47.54** | 55.61 | 58.50 | 54.07 |
| ImageNet Deblurring | FID↓ | **44.82** | 59.09 | 63.10 | 79.43 |
| ImageNet Deblurring | DISTS↓ | **0.07** | 0.10 | 0.11 | 0.15 |

RED-Diff shows slightly higher PSNR in CelebA inpainting (17.96 vs 17.70), consistent with its MAP-like over-smoothing behavior. However, Ours dominates in perceptual metrics (LPIPS/FID/DISTS), particularly in deblurring where DISTS is halved.

### Ablation Study
Correction steps for CelebA inpainting (LPIPS↓):

| Corrector | 1 Step | 5 Steps | 8 Steps |
|-----------|--------|---------|---------|
| ULA | 0.093 | 0.093 | 0.093 |
| MALA | 0.093 | 0.091 | **0.089** |

A-CSM ablation (see §4.3 blind tasks): A pure A-DSM model without A-CSM fails to estimate box size and $\sigma_1$, confirming that normalization independence across $\boldsymbol{\Sigma}$ is the root cause of successful blind estimation.

### Key Findings
- **Energy and Sample Quality Calibration**: Evaluating samples from DPS, RED-Diff, and Ours using the same $U_\theta$ reveals that DPS samples have significantly lower prior probability (biased toward OOD areas by likelihood approximation), while RED-Diff samples have high prior but low posterior probability (over-smoothed). Only Ours produces samples close to GT in both prior and posterior, making log-probability a computable metric for sampler diagnosis.
- **Energy-Guided Adaptive Scheduling**: On MNIST with $k$-pixel reconstruction, energy-guided scheduling $\boldsymbol{\delta}\boldsymbol{\Sigma}_t\propto\boldsymbol{\Sigma}_t\nabla_{\boldsymbol{\Sigma}}U_\theta\boldsymbol{\Sigma}_t$ (steepest descent in Bregman geometry) yields lower classification error than geometric scheduling at low $k$. Both converge at $k \approx 300$. On CelebA inpainting, it currently trails fixed scheduling, likely due to limited degrees of freedom in diagonal covariance.
- **MALA Unbiased Correction**: MALA requires the acceptance ratio $p(\mathbf{x}'|\boldsymbol{\Sigma}_T,\mathbf{y})/p(\mathbf{x}|\boldsymbol{\Sigma}_T,\mathbf{y})$, which pure score models cannot compute. Ours reduces LPIPS from 0.093 to 0.089 with 8 MALA steps, whereas increasing ULA steps shows no benefit, proving this is an "irreplaceable capability."
- **Blind Estimation via $\arg\max_{\boldsymbol{\Sigma}}\log p_\theta(\mathbf{y}|\boldsymbol{\Sigma})$**: In inpainting tasks with unknown box-size and $\sigma_1$, the log-probability surface shows a clear unimodal peak at the ground truth. This allows direct estimation of degradation parameters, which is impossible for Bayesian schools (no $\log p(\mathbf{y}|\boldsymbol{\Sigma})$) or Regression schools (requiring retraining for each $\boldsymbol{\Sigma}$).

## Highlights & Insights
- **Perspective Shift: "Linear Degradation ≡ Anisotropic Noise"**: Rewriting the problem via $\mathbf{H}^{-1}\mathbf{y}$ unifies various inverse problems into a single denoising framework. While the math is straightforward, it reduces the problem dimensionality to modeling a family of $\boldsymbol{\Sigma}$, serving as the core pivot of the paper.
- **A-CSM as Implicit Fokker-Planck Enforcement**: Turning the hard physical constraint of "normalization constant invariance across $\boldsymbol{\Sigma}$" into a differentiable training loss. This "regularization by conservation equations" is akin to PINNs but avoids explicit second derivatives and can be transferred to any conditional density modeling.
- **Energy as an Evaluation Metric**: Previously, comparing samplers for inverse problems relied on PSNR/LPIPS. Now, one can plot $\log p(\hat{\mathbf{x}})$ histograms to diagnose whether a sampler is biased toward the prior or the posterior—a powerful "microscope" tool showcased in §4.1.
- **Diagonal Limitation ≠ Low Expressivity**: Selecting spatial-diagonal and spectral-diagonal families covers over 90% of common linear inverse problems. Compressing $O(d^2)$ degrees of freedom down to $O(d)$ is a highly pragmatic engineering choice.

## Limitations & Future Work
- Dual score matching is more expensive to train due to the additional backprop for $\nabla_{\boldsymbol{\Sigma}}U_\theta$; future work could use sliced score matching or forward-mode JVP for acceleration.
- Covariance only supports spatial/spectral diagonals, failing to represent arbitrary $\boldsymbol{\Sigma}$ like rotational or spatially-varying blur.
- Experiments are limited to $192\times192$ resolution (AFHQ-Cat); performance scaling to $256+$ or $1024$ and numerical stability of EBM training in higher dimensions remains unproven.
- The weighting for $1/d$ and $1/d^2$ in the objective lacks an ablation, making the sensitivity of results to these ratios unknown.
- Adding a brand-new degradation operator (not in the training set $p(\boldsymbol{\Sigma})$) still requires extra training to extend the range of $\log p(\mathbf{y}|\boldsymbol{\Sigma})$, slightly compromising the Bayesian promise of "train once, solve any problem."

## Related Work & Insights
- **vs DPS / DAPS (Bayesian)**: These rely on Gaussian approximations of $p(\mathbf{y}|\mathbf{x}_t)$ for guidance. Ours learns a normalized $p(\mathbf{y}|\boldsymbol{\Sigma})$ directly, avoiding OOD bias from likelihood approximations at the cost of needing to define a covariance family $p(\boldsymbol{\Sigma})$ beforehand.
- **vs Palette / InDI (Regression)**: They train a different model for each degradation. Ours uses one model for a whole family of $\boldsymbol{\Sigma}$ and supports blind estimation. Architectural overhead is similar as covariance embeddings reuse EDM modulation layers.
- **vs Guth 2025 / Yu 2025 (Isotropic EBM)**: Their dual/time score matching only works on the 1D manifold of $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$. This work extends it to arbitrary (restricted) covariance matrices, representing a true "isotropic → anisotropic" leap.
- **vs Du 2023 / Thornton 2025 (Compositional EBM)**: While they use unnormalized energy for composition, this work emphasizes "normalization," enabling MALA, blind estimation, and probability comparisons—capabilities the compositional school lacks.
- **Insight**: Formulating "inverse problems" as "conditional density family modeling" plus "Fokker-Planck conservation constraints" is a recipe that can theoretically be applied to non-linear problems, cross-modal generation, or physics-constrained scientific inversion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Solving Inverse Problems with Flow-based Models via Model Predictive Control](solving_inverse_problems_with_flow-based_models_via_model_predictive_control.md)
- [\[CVPR 2026\] PnP-CM: Consistency Models as Plug-and-Play Priors for Inverse Problems](../../CVPR2026/image_restoration/pnp-cm_consistency_models_as_plug-and-play_priors_for_inverse_problems.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[CVPR 2026\] Dual Ascent Diffusion for Inverse Problems](../../CVPR2026/image_restoration/dual_ascent_diffusion_for_inverse_problems.md)
- [\[CVPR 2026\] Outlier-Robust Diffusion Solvers for Inverse Problems](../../CVPR2026/image_restoration/outlier-robust_diffusion_solvers_for_inverse_problems.md)

</div>

<!-- RELATED:END -->
