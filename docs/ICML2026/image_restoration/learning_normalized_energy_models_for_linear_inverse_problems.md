---
title: >-
  [Paper Note] Learning Normalized Energy Models for Linear Inverse Problems
description: >-
  [ICML 2026][Image Restoration][Anisotropic denoising] The authors reformulate "linear inverse problems" as "anisotropic denoising" and propose Anisotropic Covariance Score Matching (A-CSM) to train a **normalized** energ…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Anisotropic denoising"
  - "Covariance Score Matching"
  - "Normalized Energy Model"
  - "Posterior Sampling"
  - "Blind Inverse Problems"
date: 2026-05-08
content_hash: e41650c6e6a3d922
---

# Learning Normalized Energy Models for Linear Inverse Problems

**Conference**: ICML 2026  
**arXiv**: [2605.15487](https://arxiv.org/abs/2605.15487)  
**Code**: https://github.com/nzilberstein/Anisotropic-energy-Model (Available)  
**Area**: Image Restoration / Energy-Based Models / Diffusion Models / Linear Inverse Problems  
**Keywords**: Anisotropic denoising, Covariance Score Matching, Normalized Energy Model, Posterior Sampling, Blind Inverse Problems

## TL;DR
The authors reformulate "linear inverse problems" as "anisotropic denoising" and propose Anisotropic Covariance Score Matching (A-CSM) to train a **normalized** energy model $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$. A single model can handle inpainting, deblurring, and super-resolution while unlocking three new capabilities: energy-guided adaptive scheduling, MALA unbiased correction, and blind inverse problem estimation.

## Background & Motivation

**Background**: Diffusion models have become the mainstream prior for image inverse problems (deblurring, inpainting, super-resolution). Current approaches generally fall into two camps: the Bayesian camp treats a pre-trained unconditional diffusion model as $p(\mathbf{x})$ and uses Bayes' rule to compute $\nabla\log p(\mathbf{x}_t|\mathbf{y})$ during sampling; the Regression camp directly learns the conditional score $\nabla\log p(\mathbf{x}_t|\mathbf{y})$ from paired $(\mathbf{x},\mathbf{y})$ data.

**Limitations of Prior Work**: In the Bayesian camp, the likelihood term $p(\mathbf{y}|\mathbf{x}_t)=\int p(\mathbf{y}|\mathbf{x})p(\mathbf{x}|\mathbf{x}_t)\mathrm{d}\mathbf{x}$ involves high-dimensional integration, requiring approximations like DPS that introduce sampling bias. While the Regression camp avoids this approximation, it requires retraining a model for every different degradation operator $\mathbf{H}$, losing the flexibility of prior/likelihood decoupling. More fundamentally, both are score-based, learning only gradients rather than the **density itself**, which precludes normalized log-probability comparisons for MCMC acceptance, energy-guided scheduling, or blind estimation tasks like $\arg\max_{\boldsymbol{\Sigma}} p(\mathbf{y}|\boldsymbol{\Sigma})$.

**Key Challenge**: To simultaneously achieve: (i) prior flexibility across different degradations, (ii) unbiased sampling without likelihood approximations, and (iii) explicit normalized densities. Existing EBM-with-diffusion specialized works (Du 2023, Thornton 2025) only support isotropic noise, failing to cover the **anisotropic** covariance inherent in linear inverse problems.

**Key Insight**: The authors observe that rewriting $\mathbf{y}=\mathbf{H}\mathbf{x}+\sigma\mathbf{v}$ via $\mathbf{H}^{-1}\mathbf{y}$ is equivalent to $\mathbf{y}=\mathbf{x}+\boldsymbol{\Sigma}^{1/2}\mathbf{v}'$, where $\boldsymbol{\Sigma}=\sigma^2\mathbf{H}^{-1}(\mathbf{H}^{-1})^\top$. Thus, "solving a family of linear inverse problems" is equivalent to "denoising over a family of covariances $\boldsymbol{\Sigma}$." Learning a density conditioned on $\boldsymbol{\Sigma}$ can unify all linear degradations.

**Core Idea**: Generalize the dual score matching of Guth 2025 from isotropic to anisotropic cases by adding a **covariance score** term $\nabla_{\boldsymbol{\Sigma}}U_\theta$. This term is constrained by the Fokker-Planck equation to ensure mass conservation across $\boldsymbol{\Sigma}$, thereby training an "unnormalized energy" into a "normalized energy."

## Method

### Overall Architecture
The input consists of an observation $\mathbf{y}\in\mathbb{R}^d$ and its corresponding noise covariance $\boldsymbol{\Sigma}$ (limited to pixel-diagonal or frequency-diagonal types with $d$ degrees of freedom). The output is a scalar energy $U_\theta(\mathbf{y},\boldsymbol{\Sigma})\approx -\log p(\mathbf{y}|\boldsymbol{\Sigma})$. Taking the gradient with respect to $\mathbf{y}$ yields the score $\nabla_\mathbf{y}U_\theta$, which enables denoising/reconstruction via the anisotropic Tweedie formula $\mathbb{E}[\mathbf{x}|\mathbf{y},\boldsymbol{\Sigma}]=\mathbf{y}-\boldsymbol{\Sigma}\nabla_\mathbf{y}U_\theta$. Taking the gradient with respect to $\boldsymbol{\Sigma}$ yields the covariance score, used for adaptive scheduling and blind estimation. Architecturally, $U_\theta(\mathbf{y},\boldsymbol{\Sigma})=\tfrac{1}{2}\langle\mathbf{y},\mathbf{s}_\theta(\mathbf{y},\boldsymbol{\Sigma})\rangle$, using an EDM (Karras 2022) UNet backbone. Covariance is injected into each layer's gain modulation $\mathbf{x}_\ell\leftarrow\mathrm{SiLU}(\mathbf{x}_\ell\odot(1+\mathbf{e}_\ell))$ through newly designed spatial/spectral dual-branch embeddings. Training involves mixed sampling of spatial covariances (center/horizontal boxes, size 1~64) and spectral covariances (Gaussian deblur / 4× SR).

### Key Designs

1.  **Anisotropic Denoising Score Matching (A-DSM)**:
    - **Function**: Transforms the energy model into a "covariance-aware denoiser" using the anisotropic Tweedie formula, ensuring $\nabla_\mathbf{y}U_\theta$ approximates $\nabla_\mathbf{y}\log p(\mathbf{y}|\boldsymbol{\Sigma})$.
    - **Mechanism**: The loss is defined as $\ell_{\text{A-DSM}}=\mathbb{E}[\|\boldsymbol{\Sigma}^{1/2}\nabla_\mathbf{y}U_\theta(\mathbf{y},\boldsymbol{\Sigma})-\boldsymbol{\Sigma}^{-1/2}(\mathbf{y}-\mathbf{x})\|^2]$. Re-weighting with $\boldsymbol{\Sigma}^{1/2}$ on both sides makes the loss scale-invariant, representing an anisotropic generalization of maximum-likelihood weighting.
    - **Design Motivation**: Standard DSM assumes $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$. For tasks like inpainting where noise magnitudes vary by orders of magnitude across directions, gradients can explode or collapse; scale-invariant re-weighting enables stable training across a wide noise variance range $[10^{-9}, 10^3]$, which is essential for any-order generation and blind estimation.

2.  **Anisotropic Covariance Score Matching (A-CSM, Novel Contribution)**:
    - **Function**: Supervises the energy gradient w.r.t. $\boldsymbol{\Sigma}$, $\nabla_{\boldsymbol{\Sigma}}U_\theta$, ensuring energy values across different $\boldsymbol{\Sigma}$ are compatible up to a single constant, thereby supporting normalization.
    - **Mechanism**: The authors derive the covariance-version Tweedie identity $\nabla_{\boldsymbol{\Sigma}}U(\mathbf{y},\boldsymbol{\Sigma})=\mathbb{E}[\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}-\tfrac{1}{2}\boldsymbol{\Sigma}^{-1}(\mathbf{y}-\mathbf{x})(\mathbf{y}-\mathbf{x})^\top\boldsymbol{\Sigma}^{-1}]$ and apply a scale-invariant loss $\ell_{\text{A-CSM}}$ under the Frobenius norm. The total objective is $\tfrac{1}{d}\ell_{\text{A-DSM}}+\tfrac{1}{d^2}\ell_{\text{A-CSM}}$. After training, the model is re-normalized using $U_\theta(\mathbf{y},\boldsymbol{\Sigma})-\mathbb{E}_\mathbf{y}[U_\theta|\boldsymbol{\Sigma}]+\tfrac{1}{2}\log\det(2\pi e\boldsymbol{\Sigma})$, using $\mathbf{y}\sim\mathcal{N}(0,\boldsymbol{\Sigma})$ at large $\boldsymbol{\Sigma}$ as an anchor.
    - **Design Motivation**: This essentially uses the Fokker-Planck continuity equation $\nabla_{\boldsymbol{\Sigma}}p(\mathbf{y}|\boldsymbol{\Sigma})=\tfrac{1}{2}\nabla_\mathbf{y}^2 p(\mathbf{y}|\boldsymbol{\Sigma})$ to enforce consistency among all marginal densities so that the "constant term does not depend on $\boldsymbol{\Sigma}$." This is the fundamental extension over isotropic versions (Guth 2025, Yu 2025) and is the only way to calculate probability ratios like $p(\mathbf{y}|\boldsymbol{\Sigma}_T,\mathbf{y})/p(\mathbf{y}|\boldsymbol{\Sigma}_t,\mathbf{y})$ for MALA and blind estimation.

3.  **Dual-domain Covariance Embedding (spatial + spectral)**:
    - **Function**: Simultaneously supports pixel-domain diagonal covariance (inpainting) and frequency-domain diagonal covariance (deblurring, super-resolution) in a single UNet, compressing the $d(d-1)/2$ degrees of freedom of a covariance matrix into a $d$-dimensional vector for injection.
    - **Mechanism**: Spatial covariance is represented as a spatially varying noise map $\mathbf{e}_\ell\in\mathbb{R}^{c_\ell\times d_\ell}$, while spectral covariance is represented as channel-only modulation $\mathbf{e}_\ell\in\mathbb{R}^{c_\ell}$. Both branches compute embeddings in parallel and are injected via $\mathbf{x}_\ell\leftarrow\mathrm{SiLU}(\mathbf{x}_\ell\odot(1+\mathbf{e}_\ell))$, compatible with EDM's native isotropic channel modulation with negligible extra computation.
    - **Design Motivation**: A fully general $\boldsymbol{\Sigma}$ would lead to memory explosion at $d=4096$, but spatial-diagonal covers inpainting and frequency-diagonal covers deblur/SR without losing practical expressive power; maintaining consistency with EDM’s gain-modulation ensures inheriting the inductive bias of existing score architectures.

### Loss & Training
The total loss is $\mathcal{L}=\tfrac{1}{d}\ell_{\text{A-DSM}}+\tfrac{1}{d^2}\ell_{\text{A-CSM}}$. Training covariances are sampled with 0.5/0.5 probability between spatial types (center/horizontal boxes, size 1~64) and spectral types (Gaussian deblur kernel size 8×8, $\sigma_g=0.8$; 4× SR). The backbone is an EDM UNet. All methods (including baselines) use the same architecture, differing only in $p(\boldsymbol{\Sigma})$ and input: the Bayesian baseline uses $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$, and Palette stacks the measurement with the noisy image. Sampling uses up to 1000 NFE (1200 for CelebA inpainting).

## Key Experimental Results

### Main Results
Evaluation on CelebA 64×64 and ImageNet 64×64 for inpainting (center 45×45 box, $\sigma=10^{-4}$) and Gaussian deblurring (8×8 kernel, $\sigma=10^{-2}$), comparing against DPS, RED-Diff, DAPS, and Palette.

| Dataset / Task | Metric | Ours | DPS | RED-Diff | DAPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CelebA Inpainting | LPIPS↓ | **0.093** | 0.110 | 0.100 | 0.098 |
| CelebA Inpainting | FID↓ | **34.57** | 36.76 | 47.82 | 45.76 |
| CelebA Deblurring | LPIPS↓ | **0.002** | 0.004 | 0.006 | 0.005 |
| CelebA Deblurring | DISTS↓ | **0.04** | 0.08 | 0.08 | 0.10 |
| ImageNet Inpainting | FID↓ | **47.54** | 55.61 | 58.50 | 54.07 |
| ImageNet Deblurring | FID↓ | **44.82** | 59.09 | 63.10 | 79.43 |
| ImageNet Deblurring | DISTS↓ | **0.07** | 0.10 | 0.11 | 0.15 |

RED-Diff achieved slightly higher PSNR on CelebA inpainting (17.96 vs 17.70), consistent with its MAP-like over-smoothing behavior; however, the proposed method dominates in perceptual quality (LPIPS/FID/DISTS), particularly in deblurring where DISTS is halved.

### Ablation Study
ULA vs. MALA correction steps for CelebA inpainting (LPIPS↓):

| Corrector | 1 Step | 5 Steps | 8 Steps |
| :--- | :--- | :--- | :--- |
| ULA | 0.093 | 0.093 | 0.093 |
| MALA | 0.093 | 0.091 | **0.089** |

A-CSM ablation in §4.3 blind tasks: A pure A-DSM model without A-CSM fails completely to locate box size and $\sigma_1$, verifying that covariance-independent normalization constants are the root cause for feasible blind estimation.

### Key Findings
- **Energy and Sample Quality Calibration**: Evaluating samples from DPS/RED-Diff/Ours with $U_\theta$ under fixed observations—DPS samples show significantly lower prior probability than GT (biased OOD by likelihood approximation), and RED-Diff samples have high prior but low posterior (over-smooth). Only the proposed method's samples align with GT in both prior and posterior, making log-probabilities a computable metric for sampler diagnostics.
- **Energy-guided Adaptive Scheduling**: On MNIST random $k$-pixel reconstruction, energy-guided scheduling $\boldsymbol{\delta}\boldsymbol{\Sigma}_t\propto\boldsymbol{\Sigma}_t\nabla_{\boldsymbol{\Sigma}}U_\theta\boldsymbol{\Sigma}_t$ (steepest descent in Bregman geometry) consistently yields lower classification error than geometric scheduling at low $k$.
- **Unbiased MALA Correction**: MALA requires calculating $p(\mathbf{x}'|\boldsymbol{\Sigma}_T,\mathbf{y})/p(\mathbf{x}|\boldsymbol{\Sigma}_T,\mathbf{y})$, which pure score models cannot do. This model reduces LPIPS from 0.093 to 0.089 using 8 MALA steps, whereas increasing ULA steps is ineffective.
- **Blind Estimation via $\arg\max_{\boldsymbol{\Sigma}}\log p_\theta(\mathbf{y}|\boldsymbol{\Sigma})$**: For inpainting with unknown box-size and $\sigma_1$, the log-probability surface shows a clear unimodal peak at the ground truth, allowing direct estimation of degradation parameters—a feat impossible for both Bayesian (no $\log p(\mathbf{y}|\boldsymbol{\Sigma})$) and regression (requires retraining per $\boldsymbol{\Sigma}$) methods.

## Highlights & Insights
- **Perspective shift**: Reformulating linear inverse problems as anisotropic noise allows a unified denoising framework. It reduces the problem dimension by modeling only a family of $\boldsymbol{\Sigma}$.
- **A-CSM as implicit Fokker-Planck enforcement**: Turning the physical constraint of "constant normalization across $\boldsymbol{\Sigma}$" into a differentiable loss (similar to PINNs but avoiding explicit second derivatives) is a powerful regularization strategy transferable to any conditional density modeling.
- **Energy as an evaluation metric**: Instead of just PSNR/LPIPS, $\log p(\hat{\mathbf{x}})$ histograms can diagnose whether a sampler is OOD or over-smooth, providing a "microscope" tool for inverse problems.
- **Diagonal restriction as a pragmatic choice**: Selecting spatial-diagonal and spectral-diagonal families covers >90% of common linear inverse problems while compressing $O(d^2)$ degrees of freedom into $O(d)$.

## Limitations & Future Work
- Dual score matching training is more expensive than pure score models due to the extra backprop for $\nabla_{\boldsymbol{\Sigma}}U_\theta$; future work could use sliced score matching or forward-mode JVP for acceleration.
- The diagonal covariance restriction cannot represent arbitrary $\boldsymbol{\Sigma}$ (e.g., rotational blur, spatially varying blur), likely explaining why energy-guided scheduling underperformed on CelebA inpainting.
- Experiments reach only 192×192; scaling to 256+ or 1024 remains unproven, and EBM numerical stability poses risks in higher dimensions.
- The $1/d$ and $1/d^2$ weighting lacks detailed ablation, leaving the sensitivity of A-DSM/A-CSM balance unknown.
- Adding new degradation operators outside the training $p(\boldsymbol{\Sigma})$ still requires additional training to extend the $\log p(\mathbf{y}|\boldsymbol{\Sigma})$ range, unlike the Bayesian promise of "train once for all."

## Related Work & Insights
- **vs. DPS / DAPS (Bayesian)**: They rely on Gaussian approximations of $p(\mathbf{y}|\mathbf{x}_t)$ for guidance; this work learns the normalized $p(\mathbf{y}|\boldsymbol{\Sigma})$ directly to avoid OOD bias, at the cost of needing the covariance family $p(\boldsymbol{\Sigma})$ in advance.
- **vs. Palette / InDI (Regression)**: They train conditional models per degradation; this work covers a family of $\boldsymbol{\Sigma}$ with one model and supports blind estimation with similar architectural overhead.
- **vs. Guth 2025 / Yu 2025 / Plainer 2025 (Isotropic EBM)**: Their dual/time score matching only normalizes on the 1D manifold $\boldsymbol{\Sigma}=\sigma^2\mathbf{I}$; this work extends this to arbitrary (restricted) covariance matrices.
- **vs. Du 2023 / Thornton 2025 (Compositional EBM-diffusion)**: They focus on compositional generation via unnormalized energy; this work emphasizes "normalization," enabling MALA, blind estimation, and probability comparisons.
- **Insight**: The recipe of reformulating "inverse problems" as "conditional density family modeling" with "Fokker-Planck conservation constraints" can be ported to non-linear problems, cross-modal generation, or scientific inversion with physical constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] UOTIP: Unbalanced Optimal Transport Mapping for Unpaired Inverse Problems](uotip_unbalanced_optimal_transport_map_for_unpaired_inverse_problems.md)
- [\[ICML 2026\] Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](triadic_dynamics_aware_diffusion_posterior_sampling_for_inverse_problems_optimiz.md)
- [\[CVPR 2026\] Variational Garrote for Sparse Inverse Problems](../../CVPR2026/image_restoration/variational_garrote_for_sparse_inverse_problems.md)
- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](../../NeurIPS2025/image_restoration/learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](../../CVPR2026/image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)

</div>

<!-- RELATED:END -->
